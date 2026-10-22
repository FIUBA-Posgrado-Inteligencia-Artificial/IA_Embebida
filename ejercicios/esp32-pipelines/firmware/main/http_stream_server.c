#include "http_stream_server.h"
#include "board.h"

#if FEATURE_WEBSERVER

#include "camera.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "capture_state.h"
#include "trigger.h"
#include "model_data.h"
#include "cJSON.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "http_stream";

static QueueHandle_t s_trigger_queue;

#define BOUNDARY    "ESPFRAME"
#define PART_HEADER "--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n"

static const char INDEX_HTML[] =
"<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-CAM</title><style>"
"body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:1em;text-align:center}"
"img{max-width:100%;border:2px solid #444;border-radius:4px}"
".row{display:flex;flex-wrap:wrap;gap:1em;justify-content:center}"
".col{flex:1 1 320px;max-width:640px}"
"table{margin:0 auto;border-collapse:collapse}td,th{padding:2px 8px;text-align:left}"
"button{padding:.5em 1.5em;font-size:1em;margin:.5em}"
"#err{color:#f66}"
"</style></head><body>"
"<h3>ESP32-CAM — inferencia</h3>"
"<p style='color:#888;font-size:.85em'>Vista previa = ultima captura (activar "
"Continuo para ~3 fps). El stream MJPEG en vivo (/stream) existe pero bloquea "
"el resto de la web y las capturas mientras hay un cliente conectado.</p>"
"<div class='row'>"
"<div class='col'><h4>Ultima captura</h4><img id='last' alt='(sin captura)'>"
"<div><button onclick='trig()'>Capturar</button>"
"<label><input type='checkbox' id='cont' onchange='mode()'> Continuo</label></div>"
"<div id='err'></div><table id='res'></table></div>"
"</div><script>"
"let seq=0;"
"function trig(){fetch('/trigger',{method:'POST'})}"
"function mode(){fetch('/mode',{method:'POST',body:'continuous='+(cont.checked?1:0)})}"
"function render(r){"
"const name=i=>((r.class_names&&r.class_names[i])||('#'+i));"
"const top3=r.probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0,3);"
"let h='<tr><th>captura #</th><td>'+r.seq+'</td></tr>';"
"h+='<tr><th>top3</th><td>'+top3.map(([p,i])=>name(i)+' '+(p*100).toFixed(1)+'%').join(' | ')+'</td></tr>';"
"h+='<tr><th>captura</th><td>'+r.t_capture_ms+' ms</td></tr>';"
"h+='<tr><th>preproceso</th><td>'+r.t_preprocess_ms+' ms</td></tr>';"
"h+='<tr><th>inferencia</th><td>'+r.t_inference_ms+' ms</td></tr>';"
"h+='<tr><th>total</th><td>'+(r.t_capture_ms+r.t_preprocess_ms+r.t_inference_ms)+' ms</td></tr>';"
"if(r.fps>0)h+='<tr><th>fps</th><td>'+r.fps.toFixed(1)+'</td></tr>';"
"r.probs.forEach((p,i)=>{h+='<tr><td>'+((r.class_names&&r.class_names[i])||('#'+i))+'</td>"
"<td>'+(p*100).toFixed(1)+'%</td></tr>';});"
"res.innerHTML=h;}"
"async function poll(){try{"
"const r=await(await fetch('/result')).json();"
"cont.checked=(r.mode=='continuous');"
"err.textContent=r.last_error||'';"
/* imagen y tabla del MISMO seq, y juntas: la tabla se pinta recién cuando
 * la imagen nueva terminó de bajar */
"if(r.seq>0&&r.seq!=seq){seq=r.seq;"
"const im=new Image();"
"im.onload=()=>{last.src=im.src;render(r);};"
"im.onerror=()=>{render(r);};"
"im.src='/last.jpg?'+r.seq;}"
"}catch(e){}setTimeout(poll,500)}"
"poll();"
"</script></body></html>";

static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, sizeof(INDEX_HTML) - 1);
}

#if BOARD_HAS_CAMERA
static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = httpd_resp_set_type(req,
        "multipart/x-mixed-replace;boundary=" BOUNDARY);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char part_hdr[96];
    ESP_LOGI(TAG, "stream client connected");

    while (true) {
        camera_fb_t *fb = NULL;
        if (camera_capture_jpeg(&fb) != ESP_OK || fb == NULL) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        int n = snprintf(part_hdr, sizeof part_hdr, PART_HEADER, fb->len);
        res = httpd_resp_send_chunk(req, part_hdr, n);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, "\r\n", 2);

        camera_return(fb);

        if (res != ESP_OK) break;  /* cliente desconectado */
    }

    ESP_LOGI(TAG, "stream client disconnected");
    return res;
}
#endif  /* BOARD_HAS_CAMERA */

static esp_err_t trigger_handler(httpd_req_t *req)
{
    trigger_source_t src = TRIGGER_HTTP;
    xQueueSend(s_trigger_queue, &src, 0);
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t mode_handler(httpd_req_t *req)
{
    char body[32] = {0};
    int n = httpd_req_recv(req, body, sizeof body - 1);
    if (n <= 0) { httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "empty body"); return ESP_FAIL; }
    /* body esperado: continuous=0 | continuous=1 */
    const char *p = strstr(body, "continuous=");
    if (!p) { httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "missing continuous="); return ESP_FAIL; }
    capture_state_set_mode(p[11] == '1' ? CAPTURE_MODE_CONTINUOUS : CAPTURE_MODE_SINGLE);
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t last_jpg_handler(httpd_req_t *req)
{
    const uint8_t *buf; size_t len;
    capture_state_borrow_jpeg(&buf, &len);
    if (buf == NULL) {
        capture_state_release_jpeg();
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "no capture yet");
        return ESP_FAIL;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t res = httpd_resp_send(req, (const char *)buf, len);
    capture_state_release_jpeg();
    return res;
}

static esp_err_t result_handler(httpd_req_t *req)
{
    capture_status_t st;
    capture_state_get_status(&st);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "seq", st.seq);
    cJSON_AddStringToObject(root, "mode",
        st.mode == CAPTURE_MODE_CONTINUOUS ? "continuous" : "single");
    cJSON_AddStringToObject(root, "last_error", st.last_error);
    if (st.seq > 0) {
        if (st.result.top1_idx >= 0 && st.result.top1_idx < MODEL_NUM_CLASSES) {
            cJSON_AddStringToObject(root, "top1", MODEL_CLASSES[st.result.top1_idx]);
        } else {
            cJSON_AddStringToObject(root, "top1", "unknown");
        }
        cJSON_AddNumberToObject(root, "top1_idx", st.result.top1_idx);
        cJSON_AddNumberToObject(root, "top1_confidence", st.result.top1_confidence);
        cJSON *probs = cJSON_AddArrayToObject(root, "probs");
        for (int i = 0; i < MODEL_NUM_CLASSES; i++)
            cJSON_AddItemToArray(probs, cJSON_CreateNumber(st.result.probs[i]));
        cJSON_AddNumberToObject(root, "t_capture_ms", st.t_capture_ms);
        cJSON_AddNumberToObject(root, "t_preprocess_ms", st.t_preprocess_ms);
        cJSON_AddNumberToObject(root, "t_inference_ms", st.t_inference_ms);
        cJSON_AddNumberToObject(root, "fps", st.fps);
        cJSON *names = cJSON_AddArrayToObject(root, "class_names");
        for (int i = 0; i < MODEL_NUM_CLASSES; i++)
            cJSON_AddItemToArray(names, cJSON_CreateString(MODEL_CLASSES[i]));
    }
    char *out = cJSON_PrintUnformatted(root);
    if (out == NULL) {
        cJSON_Delete(root);
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "oom");
    }
    httpd_resp_set_type(req, "application/json");
    esp_err_t res = httpd_resp_send(req, out, HTTPD_RESP_USE_STRLEN);
    cJSON_free(out);
    cJSON_Delete(root);
    return res;
}

esp_err_t http_stream_server_start(QueueHandle_t trigger_queue)
{
    s_trigger_queue = trigger_queue;
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port       = HTTP_PORT;
    /* ctrl_port debe ser distinto de server_port */
    cfg.ctrl_port         = (uint16_t)(HTTP_PORT + 1);
    cfg.recv_wait_timeout = 10;
    cfg.send_wait_timeout = 10;
    cfg.lru_purge_enable  = true;
    cfg.max_uri_handlers  = 8;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "start failed: %s", esp_err_to_name(err));
        return err;
    }

    const httpd_uri_t index_uri  = { .uri = "/",       .method = HTTP_GET, .handler = index_handler };
    httpd_register_uri_handler(server, &index_uri);
#if BOARD_HAS_CAMERA
    const httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler };
    httpd_register_uri_handler(server, &stream_uri);
#else
    ESP_LOGW(TAG, "/stream no registrado — cámara deshabilitada");
#endif

    const httpd_uri_t trigger_uri = { .uri = "/trigger",  .method = HTTP_POST, .handler = trigger_handler };
    const httpd_uri_t mode_uri    = { .uri = "/mode",     .method = HTTP_POST, .handler = mode_handler };
    const httpd_uri_t last_uri    = { .uri = "/last.jpg", .method = HTTP_GET,  .handler = last_jpg_handler };
    const httpd_uri_t result_uri  = { .uri = "/result",   .method = HTTP_GET,  .handler = result_handler };
    httpd_register_uri_handler(server, &trigger_uri);
    httpd_register_uri_handler(server, &mode_uri);
    httpd_register_uri_handler(server, &last_uri);
    httpd_register_uri_handler(server, &result_uri);

    ESP_LOGI(TAG, "HTTP server started on port %d  →  http://<IP>/",
             HTTP_PORT);
    return ESP_OK;
}

#else

esp_err_t http_stream_server_start(QueueHandle_t trigger_queue) { (void)trigger_queue; return ESP_OK; }

#endif

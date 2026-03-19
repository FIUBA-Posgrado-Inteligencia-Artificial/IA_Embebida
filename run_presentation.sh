#!/bin/bash

# Puerto en el que se levantará el servidor web (buscar uno libre)
PORT=8000
while :; do
    # Probar si el puerto está libre usando Python
    if python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('127.0.0.1', $PORT)); s.close()" 2>/dev/null; then
        break
    fi
    echo "El puerto $PORT ya está en uso. Intentando con el siguiente..."
    PORT=$((PORT + 1))
done

echo "Iniciando servidor local en el puerto $PORT..."

# Levantar el servidor HTTP de Python en segundo plano
python3 -m http.server $PORT &
SERVER_PID=$!

# Esperar 1 segundo para asegurarse de que el servidor esté listo
sleep 1

# Abrir Google Chrome en la URL de las clases
echo "Abriendo la presentación en Google Chrome..."
google-chrome "http://localhost:${PORT}/clases/clases.html" 2>/dev/null || xdg-open "http://localhost:${PORT}/clases/clases.html"

echo "Presiona Ctrl+C para detener el servidor."

# Asegurarse de cerrar el servidor de Python cuando se detenga el script
trap "echo 'Deteniendo el servidor...'; kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

# Mantener el script en ejecución para mantener vivo el servidor
wait $SERVER_PID

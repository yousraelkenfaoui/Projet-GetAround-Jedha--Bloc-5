# Utilisez l'image Python 3.8 comme base
FROM python:3.12.4

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/
 
# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le fichier api.py dans le conteneur
COPY api.py /app/

# Exposez le port que l'application utilisera
EXPOSE 5000

# Commande pour démarrer l'application
CMD gunicorn api:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker


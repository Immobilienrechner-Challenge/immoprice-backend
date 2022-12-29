pip install -r requirements.txt
python manage.py migrate
gunicorn api.wsgi -b :8099
all: flask celery flower

flask:
	flask --debug --app views run

celery:
	celery -A views.celery_worker worker --loglevel=info

flower:
	celery -A views.celery_worker flower --port=9000

celery_clear_queue:
	celery -A views.celery_worker purge


stress:
	locust -f tests/locustfile.py  --headless --users=100  -r=30  --host=http://localhost:5000   --run-time=180s  --csv=log-test.csv

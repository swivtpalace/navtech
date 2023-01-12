from locust import HttpUser, task, between

class UserTraffic(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_work_load(self):
        response = self.client.get('/test-workload')
        return response



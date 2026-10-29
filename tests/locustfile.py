from locust import HttpUser, task, between


class LegalAssistantUser(HttpUser):
    wait_time = between(6.5, 10)

    def on_start(self):
        # Locust keeps a separate real cookie jar for each simulated browser.
        session = self.client.get('/session')
        session.raise_for_status()
        self.headers = {'X-Session-CSRF': session.json()['csrf_token']}
        response = self.client.post('/start-conversation',
            json={'title': 'Concurrency check'}, headers=self.headers)
        self.conversation_id = response.json().get('conversation_id') if response.ok else None

    @task(3)
    def chat_with_bot(self):
        if self.conversation_id:
            self.client.post(f'/chat/{self.conversation_id}',
                json={'user_prompt': 'Thủ tục đăng ký khai sinh cần những giấy tờ gì?'},
                headers=self.headers, name='/chat/[id]')

    @task(1)
    def view_history(self):
        self.client.get('/conversations')

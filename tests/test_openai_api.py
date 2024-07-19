import unittest
from src.openai_api import get_completion, set_openai_key

class TestGetCompletion(unittest.TestCase):

    # setup function for tests
    
    def setUp(self) -> None:
        set_openai_key()
    
    
    def test_get_completion_with_default_model(self):
        prompt = "What is the capital of France?"
        response, finish_reason = get_completion(prompt)
        self.assertIn("Paris", response)
        self.assertEqual(finish_reason, "stop")

    def test_get_completion_with_gpt3turbo_0125(self):
        prompt = "Generate a plan to solve the following problem: How to create a new business idea in 30 words?"
        response, finish_reason = get_completion(prompt, model="gpt-3.5-turbo-0125")
        self.assertNotEqual(response, "")
        self.assertEqual(finish_reason, "stop")
        
    def test_get_completion_with_gpt4(self):
        prompt = "Generate a plan to solve the following problem: How to create a new business idea in 30 words?"
        response, finish_reason = get_completion(prompt, model="gpt-4")
        self.assertNotEqual(response, "")
        self.assertEqual(finish_reason, "stop")

    def test_get_completion_with_gpt4omini(self):
        prompt = "Generate a plan to solve the following problem: How to create a new business idea in 30 words?"
        response, finish_reason = get_completion(prompt, model="gpt-4o-mini")
        self.assertNotEqual(response, "")
        self.assertEqual(finish_reason, "stop")
        
    def test_get_completion_with_temperature_with_default_model(self):
        prompt = "Write a short story about a sunny day in 20 words."
        response1, _ = get_completion(prompt, temperature=0)
        response2, _ = get_completion(prompt, temperature=1)
        self.assertNotEqual(response1, response2)

    def test_get_completion_with_invalid_model(self):
        prompt = "What is the speed of light?"
        response, err = get_completion(prompt, model="invalid-model")
        self.assertEqual(response, 'model_not_found')
        self.assertEqual(err, 'openai_exception')


if __name__ == '__main__':
    unittest.main()
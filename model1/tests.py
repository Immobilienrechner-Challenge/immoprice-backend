from django.test import TestCase


class ModelTest(TestCase):
    def test(self):
        self.assertEqual("lmao", "lmao")

    def test2(self):
        self.assertEqual("rofl", "rofl")

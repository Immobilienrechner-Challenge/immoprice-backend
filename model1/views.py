import json

from django.http import HttpResponse

def index(request):
    return HttpResponse(f"\"request.GET\": {json.dumps(request.GET)}")

def test(request, testnum):
    return HttpResponse(f"Model 1 test. Value: {testnum}")

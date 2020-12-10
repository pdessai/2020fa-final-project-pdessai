from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import View

from rest_framework.views import APIView
from rest_framework.response import Response

# import numpy as np
import pandas as pd


User = get_user_model()

class HomeView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'charts.html', {"customers": 10})



def get_data(request, *args, **kwargs):
    data = {
        "sales": 100,
        "customers": 10,
    }
    return JsonResponse(data) # http response


class ChartData(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        qs_count = User.objects.all().count()
        labels = ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]
        arabic = pd.read_csv('src/data/names/Arabic.txt', header=None)
        chinese = pd.read_csv('src/data/names/Chinese.txt', header=None)
        czech = pd.read_csv('src/data/names/Czech.txt', header=None)

        arabic_count = pd.DataFrame(arabic.count())

        default_items = [arabic_count, 268, 519, 519, 3668, 200, 300, 566, 766, 344, 342, 677, 3000, 3555, 6777, 3444, 754, 786]
        data = {
                "labels": labels,
                "default": default_items,
        }
        return Response(data)


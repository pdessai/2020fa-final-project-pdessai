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

        arabic = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Arabic.txt', header=None)
        chinese = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Chinese.txt', header=None)
        czech = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Czech.txt', header=None)
        dutch = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Dutch.txt', header=None)
        english = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/English.txt', header=None)
        french = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/French.txt', header=None)
        german = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/German.txt', header=None)
        greek = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Greek.txt', header=None)
        irish = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Irish.txt', header=None)
        italian = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Italian.txt', header=None)
        japanese = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Japanese.txt', header=None)
        korean = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Korean.txt', header=None)
        polish = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Polish.txt', header=None)
        portuguese = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Portuguese.txt', header=None)
        # russian = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Russian.txt', header=None)
        scottish = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Scottish.txt', header=None)
        spanish = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Spanish.txt', header=None)
        vietnamese = pd.read_csv('/Users/preciladessai/Documents/2020fa-final-project-pdessai/data/names/Vietnamese.txt', header=None)


        arabic_count = pd.DataFrame(arabic).count()
        chinese_count = pd.DataFrame(chinese).count()
        czech_count = pd.DataFrame(czech).count()
        dutch_count = pd.DataFrame(dutch).count()
        english_count = pd.DataFrame(english).count()
        german_count = pd.DataFrame(german).count()
        french_count = pd.DataFrame(french).count()
        greek_count = pd.DataFrame(greek).count()
        irish_count = pd.DataFrame(irish).count()
        italian_count = pd.DataFrame(italian).count()
        japanese_count = pd.DataFrame(japanese).count()
        korean_count = pd.DataFrame(korean).count()
        polish_count = pd.DataFrame(polish).count()
        portuguese_count = pd.DataFrame(portuguese).count()
        # russian_count = pd.DataFrame(russian).count()
        scottish_count = pd.DataFrame(scottish).count()
        spanish_count = pd.DataFrame(spanish).count()
        vietnamese_count = pd.DataFrame(vietnamese).count()


        default_items = [arabic_count, chinese_count, czech_count, dutch_count, english_count, french_count,
                         german_count, greek_count, irish_count, italian_count, japanese_count, korean_count,
                         polish_count, portuguese_count, 9408, scottish_count, spanish_count, vietnamese_count]
        # default_items = [235, 22342, 234234, 23434, 4324, 4534, 234, 234, 754, 235, 356, 234, 345, 23523, 234, 234, 23, 425]
        data = {
                "labels": labels,
                "default": default_items,
        }
        return Response(data)

    # def get_output(self, request, format=None):
    #     output = pd.read_csv("/Users/preciladessai/Documents/Django-Chart.js/src/data/output/lstm_output.csv", header=True)
    #     default_items = output
    #     data = {
    #         "labels": output.head(),
    #         "default": default_items,
    #     }
    #     return Response(data)


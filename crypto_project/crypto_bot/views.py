# importing the django modules
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .rag_crypto import rag_chatbot

@csrf_exempt
def cryptoUI(request):
    # check the post method
    if request.method == "POST":
        # get the user_query
        user_query = request.POST.get("user_query", "")
        if user_query:
            # call the crypto_bot with the user_query
            # creating the instance of the crypto bot
            bot =  rag_chatbot()
            response = bot.query(user_query)
            return JsonResponse({"response": response})
    return render(request, "render.html")
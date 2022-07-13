from django.http import HttpResponse
from django.template import loader
#from django.shortcuts import render

from .models import User


def index(request):
	user_list = User.objects.order_by('id')
	
	template = loader.get_template('index.html')
	
	context = {
        'user_list': user_list,
    }
    
	return HttpResponse(template.render(context, request))
	
	#context = {'user_list': user_list}
	#return render(request, 'index.html', context)
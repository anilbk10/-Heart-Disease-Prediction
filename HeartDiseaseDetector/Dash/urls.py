from django.urls import path

from . import views

urlpatterns = [
    # Landing page with a button that takes the user to the prediction form
    path('', views.home, name='home'),
    # Page that contains the full feature form and shows the prediction result
    path('predict/', views.predict, name='predict'),
]
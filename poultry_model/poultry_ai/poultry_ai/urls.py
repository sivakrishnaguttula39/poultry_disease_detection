from django.contrib import admin
from django.urls import path
from detector.views import predict_disease
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', predict_disease),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

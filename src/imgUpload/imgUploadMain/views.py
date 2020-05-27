from django.shortcuts import render
from imgUploadMain.models import Img
# Create your views here.
def uploadImg(request):
    if request.method == 'POST':
        img = Img()
        img.img_url = request.FILES.get('img')
        img.save()
    return render(request,'imgUpload.html')

def showImg(request):
    imgs = Img.objects.all()
    context = {
        'imgs' : imgs
    }
    return render(request,'showImg.html',context)

from django.db import models

# Create your models here.
# Create your models here.
class Ticketdata(models.Model):
    ticket_number = models.CharField(max_length=50)
    L1_Tag = models.CharField(max_length=50,null=True)
    L2_Tag = models.CharField(max_length=50,null=True)
    Short_description = models.TextField(null=True)

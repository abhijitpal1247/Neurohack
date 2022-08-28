from django.db import models

# Create your models here.
# Create your models here.
class Ticketdata(models.Model):
    ticket_number = models.TextField()
    L1_Tag = models.TextField(null=True)
    L2_Tag = models.TextField(null=True)
    Short_description = models.TextField(null=True)

class Trend_data(models.Model):
    ticket_number = models.TextField()
    Created_time=models.TextField(null=True)
    Resolution_time=models.FloatField(null=True)
    

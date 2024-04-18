from wtforms import Form, FloatField

class HousePriceForm(Form):
    crime_rate = FloatField('Crime Rate')
    num_rooms = FloatField('Number of Rooms')
    indus = FloatField('Indus')
    house_age = FloatField('House Age')
    distance = FloatField('Distance')
    tax_rate = FloatField('Tax Rate Per $10,000')

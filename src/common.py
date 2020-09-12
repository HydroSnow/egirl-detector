
from requests import get
from PIL import Image
from io import BytesIO
from numpy import concatenate

def get_image_array(uuid):
    response = get(f'https://crafatar.com/avatars/{uuid}.png?size=8')
    buffer = BytesIO(response.content)
    image = Image.open(buffer)
    data = image.getdata()
    array = concatenate(data, axis=0).tolist()
    return array

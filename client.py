import requests

body = {
    'mean_radius':13.54,
    'mean_texture':14.36,
    'mean_perimeter':87.46,
    'mean_area':566.3,
    'mean_smoothness':0.09779
    }

response = requests.post(url = 'http://127.0.0.1:8000/diagnosis', json = body)
print (response.json())
# output: {'score': 0.866490130600765}

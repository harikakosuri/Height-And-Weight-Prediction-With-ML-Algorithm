import requests

url1 = 'http://localhost:5000/predict_api1'
r1= requests.post(url1,json={'Height':145})

print(r1.json())


url2 = 'http://localhost:5000/predict_api2'
r2= requests.post(url2,json={'Weight':45})

print(r2.json())


url3 = 'http://localhost:5000/predict_api3'
r3= requests.post(url3,json={'Height':145})

print(r3.json())


url4 = 'http://localhost:5000/predict_api4'
r4= requests.post(url4,json={'Weight':45})

print(r4.json())

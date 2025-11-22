import requests

url="http://localhost/attendproject/savestudent.php"


def insert_record( name ):

    param={
        "name":name
    }

    resp=requests.get( url, param )

    print(f"response :{resp}")

    #print(resp.text)

    return resp.text
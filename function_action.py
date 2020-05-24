#
#
# main() will be run when you invoke this action
#
# @param Cloud Functions actions accept a single parameter, which must be a JSON object.
#
# @return The output of this action, which must be a JSON object.
#
#
#
#
# main() will be run when you invoke this action
#
# @param Cloud Functions actions accept a single parameter, which must be a JSON object.
#
# @return The output of this action, which must be a JSON object.
#
#
import urllib.request, json 

def main(params):
    if params["district"] == "place":
        
        
        url = params.get("link")
      
        distr = params["village"]
        state = params["state"]
        if state == "Tamilnadu": 
            
            with urllib.request.urlopen(url) as url:
                
                
                data = json.loads(url.read().decode('utf-8'))
    
                active = data["Tamil Nadu"]["districtData"][distr]["active"]
                active_cases = ("active cases are\t""{}".format(active))
               
                return { 'result':  active_cases }
        if distr == "Kasargod":
            
                
            with urllib.request.urlopen(url) as url:
                data = json.loads(url.read().decode('utf-8'))
    
                active = data[state]["districtData"]["Kasaragod"]["active"]
                active_cases = ("active cases are\t""{}".format(active))
               
                return { 'result':  active_cases }
        else:
            
              
            with urllib.request.urlopen(url) as url:
                data = json.loads(url.read().decode('utf-8'))
    
                active = data[state]["districtData"][distr]["active"]
                active_cases = ("active cases are\t""{}".format(active))
               
                return { 'result':  active_cases }
            
    else:
        pass


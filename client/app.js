const API_BASE_URL = window.location.hostname
  ? `${window.location.protocol}//${window.location.hostname}:5001`
  : "http://127.0.0.1:5001"
function openOverlay(id){
document.getElementById(id).style.display="flex"
}

function closeOverlay(){
document.querySelectorAll(".overlay").forEach(o=>{
o.style.display="none"
})
}



function openUpload(){

let input=document.createElement("input")
input.type="file"
input.accept="video/*"

input.onchange=()=>{

let file=input.files[0]

analyzeVideo(file)

}

input.click()

}



function openCamera(){

openOverlay("analysis")

document.getElementById("loading").style.display="block"
document.getElementById("result").style.display="none"

setTimeout(()=>{

showFakeResult()

},3000)

}



async function analyzeVideo(file){

openOverlay("analysis")

document.getElementById("loading").style.display="block"
document.getElementById("result").style.display="none"


let form=new FormData()

form.append("video",file)


try{

let res=await fetch(`${API_BASE_URL}/predict`,{

method:"POST",
body:form

})


let data=await res.json()

showResult(data)

}

catch(err){

alert("Backend connection error")

}

}



function showResult(data){

document.getElementById("loading").style.display="none"
document.getElementById("result").style.display="block"

document.getElementById("prediction").innerText="Prediction: "+data.prediction

document.getElementById("confidence").innerText="Confidence: "+(data.confidence*100).toFixed(2)+"%"

document.getElementById("people").innerText="Persons Detected: "+data.person_count

document.getElementById("frame").src="data:image/jpeg;base64,"+data.original_frame

}



function showFakeResult(){

document.getElementById("loading").style.display="none"
document.getElementById("result").style.display="block"

document.getElementById("prediction").innerText="Prediction: Violent"

document.getElementById("confidence").innerText="Confidence: 92%"

document.getElementById("people").innerText="Persons Detected: 3"

}

const dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("header"),
button = dropArea.querySelector("button"),
icon = dropArea.querySelector(".icon");
input = document.querySelector("input");
let file;
let form; 

button.onclick = ()=>{
  input.click(); 
}

input.addEventListener("change", function(){
  file = this.files[0];
  dropArea.classList.add("active");
  showFile();
});

function isEmpty(){
    if (($('.input_form').val().length === 0) || ($('.input_file')[0].files.length === 0)) {
      return true;
    }else{
      return false;
    }
}

function upload(){
  if (!isEmpty()){
    $('#loading').show();
  }
}

function showFile(){
  let fileType = file.type; 
  let fileName = file.name;
  let validExtensions = ["image/jpeg", "image/jpg", "image/png", "image/tiff"]; //adding some valid image extensions in array

  if(validExtensions.includes(fileType)){ //if user selected file is an image file
    button.style.visibility = "hidden";
    icon.style.visibility = "hidden";
    let fileReader = new FileReader(); //creating new FileReader object

    if ((fileType == validExtensions[0]) || (fileType == validExtensions[1]) || (fileType == validExtensions[2])) {
      fileReader.onload = ()=>{
        let fileURL = fileReader.result; 
        let imgTag = `<img src="${fileURL}" alt="">`; 
        dropArea.innerHTML = imgTag;
      }
      fileReader.readAsDataURL(file);
    }else if (fileType == validExtensions[3]){
      fileReader.onload = ()=>{
        let imgName= `<p>${fileName}</p>`; 
        dropArea.innerHTML = imgName;
      }
      fileReader.readAsDataURL(file);
    }

  }else{
    alert("This is not an Image File!");
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  }
}
function changeImage() {
    var userInput = document.getElementById("user-input").value;
    var imageElement = document.getElementById("sm");
    if (userInput) {
        let s = "../dataset/"+ userInput + ".jpg";
        imageElement.style.display = "flex";
        imageElement.src = s;
    } else {
        // console.log
        imageElement.style.display = "none";
    }
}

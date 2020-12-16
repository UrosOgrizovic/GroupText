function submitForm(event) {
    var formData = new FormData(event.target.elements.file.files[0]);
//    formData.append(event.target.elements.file.files[0]);
    event.preventDefault();
    console.log(event.target.elements);
    var file = event.target.elements.file.files[0];
    console.log("A");
    console.log(file);
    console.log(JSON.stringify(file));
    $.ajax({
        type: 'POST',
        url: '/',
        data: formData,
        dataType: 'json',
//        contentType: 'application/json;charset=UTF-8',
        contentType: 'multipart/form-data',
        success: function(obj) {
            console.log(obj);

        }
    });
}
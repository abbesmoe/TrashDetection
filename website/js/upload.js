$(document).ready(function() {

    $('form').on('submit', function(event) {

        event.preventDefault();

        var formData = new FormData($('form')[0]);

        $.ajax({
            xhr : function() {
                var xhr = new window.XMLHttpRequest();

                xhr.upload.addEventListener('progress', function(e) {

                    if (e.lengthComputable) {

                        console.log('Bytes Loaded: ' + e.loaded);
                        console.log('Total Size: ' + e.total);
                        console.log('Percentage Upladed ' + (e.loaded / e.total));

                    }

                });

                return xhr;
            },
            type : 'POST',
            url : '/upload',
            data : formData,
            processData : false,
            contentType : false,
            success : function() {
                alert('File uploaded!');
            }
        });
    });

});
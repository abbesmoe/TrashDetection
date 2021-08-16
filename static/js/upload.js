$(document).ready(function() {
    $('form').on('submit', function(event) {
        var button = document.getElementById('submit');
        button.outerHTML = '<button type="button" id="submit" class="btn btn-primary" type="button" disabled><span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>Loading...</button>';
        event.preventDefault();
        var formData = new FormData($('form')[0]);
        $.ajax({
            xhr: function() {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener('progress', function(e) {
                    if (e.lengthComputable) {
                        console.log('Bytes Loaded: ' + e.loaded);
                        console.log('Total Size: ' + e.total);
                        console.log('Percentage Upladed ' + (e.loaded / e.total));
                        var percent = Math.round((e.loaded / e.total) * 100);
                        if (percent == 100) {
                            $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '% ' + '(Upload complete please wait for the detection. Thank you for patience!)');
                        } else {
                            $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%');
                        }
                    }
                });
                return xhr;
            },
            type: 'POST',
            url: '/upload',
            data: formData,
            processData: false,
            contentType: false,
            success: function() {
                button = document.getElementById('submit');
                button.outerHTML = '<input type="submit" id="submit" value="Detect and Upload" class="btn btn-info">';
                $('#progressBar').attr('aria-valuenow', 100).css('width', 100 + '%').text(100 + '% ' + 'Successfully uploaded and detected all files! please proceed to the library page to check them or to the search page to filter through the results! Thank you for your patience!');
                alert('Files have been uploaded and detected!');
            }
        });
    });
});
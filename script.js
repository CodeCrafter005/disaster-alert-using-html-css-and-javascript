document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('disasterForm').addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent form from refreshing the page

        // Get form data
        const location = document.getElementById('location').value;
        const severity = document.getElementById('severity').value;
        const phoneNumbersInput = document.getElementById('phoneNumbers').value;

        // Split phone numbers into an array
        const phoneNumbers = phoneNumbersInput.split(',').map(num => num.trim());

        // Create alert message
        const alertMessage = `🔥 FIRE ALERT: Fire detected in ${location}! Severity: ${severity.toUpperCase()}. Stay safe!`;

        // Display the alert on the webpage
        const alertList = document.getElementById('alertList');
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item';
        alertItem.textContent = alertMessage;
        alertList.appendChild(alertItem);

        // Clear the form fields
        document.getElementById('disasterForm').reset();

        // Send SMS alerts
        sendMobileAlert(alertMessage, phoneNumbers);
    });

    // Function to send mobile alerts via Twilio
    function sendMobileAlert(message, phoneNumbers) {
        console.log('Sending mobile alert:', message, phoneNumbers);

        fetch('http://localhost:3000/send-alert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message, phoneNumbers: phoneNumbers }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Alert sent successfully:', data);
        })
        .catch(error => {
            console.error('Error sending alert:', error);
        });
    }
});
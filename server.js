const express = require('express');
const bodyParser = require('body-parser');
const twilio = require('twilio');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS for all routes
app.use(cors());
app.use(bodyParser.json());

// Twilio credentials
const accountSid = 'ACba4261193184140e83571f0fe727d85e'; // Replace with your Twilio SID
const authToken = '76022b182ba7141544c8f98c957641b7'; // Replace with your Twilio Token
const twilioPhoneNumber = '+18313152626'; // Replace with your Twilio number

const client = new twilio(accountSid, authToken);

// Endpoint to send alerts
app.post('/send-alert', async (req, res) => {
    const { message, phoneNumbers } = req.body;

    if (!message || !phoneNumbers || !Array.isArray(phoneNumbers)) {
        return res.status(400).json({ success: false, message: 'Invalid request body.' });
    }

    console.log('Received alert request:', message, phoneNumbers);

    // Send SMS to each phone number
    const sendPromises = phoneNumbers.map(async (phoneNumber) => {
        try {
            const msg = await client.messages.create({
                body: message,
                from: twilioPhoneNumber,
                to: phoneNumber,
            });
            console.log(`Alert sent to ${phoneNumber}:`, msg.sid);
            return { phoneNumber, success: true, sid: msg.sid };
        } catch (error) {
            console.error(`Error sending alert to ${phoneNumber}:`, error);
            return { phoneNumber, success: false, error: error.message };
        }
    });

    const results = await Promise.all(sendPromises);
    res.json({ success: true, results });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
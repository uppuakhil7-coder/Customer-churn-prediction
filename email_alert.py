import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email(subject, message):
    sender_email = "uppuakhil7@gmail.com"
    receiver_email = "uppuakhil07@gmail.com"
    password = "djng ceha xkaj eeav"  # Use your 16-character app password

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)

        print("✅ Email alert sent successfully.")

    except Exception as e:
        print(f"❌ Error sending email: {e}")

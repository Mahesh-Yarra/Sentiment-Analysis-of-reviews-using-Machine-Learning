import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email, subject, body):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls

    sender_email = "sdiddtayade2@gmail.com"  # Your Gmail address
    password = "ijjjtfthyezvkvmj "  # Your Gmail password

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Construct the email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject

    # Attach the body of the email
    message.attach(MIMEText(body, "plain"))

    try:
        # Try to establish a connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, password)  # Log in to the account

        # Send the email
        server.sendmail(sender_email, to_email, message.as_string())
        print("Email successfully sent!")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
    finally:
        server.quit()  # Close the connection

import os, requests, smtplib
from email.message import EmailMessage

class Notifier:
    def __init__(self,config):
        self.cfg=config.get("notifications",{})
    def send(self,message,subject="Seraph Alert"):
        if not self.cfg.get("enabled",False): return
        d=self.cfg.get("discord",{})
        if d.get("enabled") and d.get("webhook_url"):
            try: requests.post(d["webhook_url"],json={"content":message},timeout=8)
            except Exception: pass
        e=self.cfg.get("email",{})
        password=os.getenv("SERAPH_EMAIL_PASSWORD")
        if e.get("enabled") and password and e.get("sender_email") and e.get("receiver_email"):
            try:
                m=EmailMessage(); m["Subject"]=subject; m["From"]=e["sender_email"]; m["To"]=e["receiver_email"]; m.set_content(message)
                with smtplib.SMTP(e.get("smtp_server","smtp.gmail.com"),int(e.get("smtp_port",587)),timeout=10) as s:
                    s.starttls(); s.login(e["sender_email"],password); s.send_message(m)
            except Exception: pass

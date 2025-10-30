import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any
import glob

logger = logging.getLogger(__name__)

class SimpleEmailService:
    """
    Simple service class for sending Excel analysis results via email.
    """
    
    def __init__(self):
        # Email configuration - uses environment variables if available, otherwise hardcoded defaults
        self.smtp_server = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', 'alexabebe114@gmail.com')
        # IMPORTANT: This must be a Gmail App Password, not your regular password!
        # Generate one at: https://myaccount.google.com/apppasswords
        self.smtp_password = os.getenv('SMTP_PASSWORD', 'evzf afnk sgbe csok')
        self.recipient_email = os.getenv('SMTP_RECIPIENT_EMAIL', 'ashish@mobifly.in')
        
    def send_excel_results(self, excel_file_path: str, company_name: str) -> bool:
        """
        Send Excel analysis results via email.
         
        Args:
            excel_file_path: Path to the Excel file to send
            company_name: Name of the company analyzed
            
        Returns:
            Boolean indicating success/failure
        """
        try:
            # Verify Excel file exists
            excel_path = Path(excel_file_path)
            if not excel_path.exists():
                logger.error(f"Excel file not found: {excel_file_path}")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Corporate Analysis Results - {company_name}"
            
            # Email body
            body = f"""
Hello,

Please find attached the corporate intelligence analysis results for {company_name}.

The Excel file contains multiple sheets:
- All_Articles: Complete analysis results
- Subsidiary_Specific: Articles specific to subsidiaries  
- Parent_Company_Impact: Articles affecting parent company
- Adverse_Only: Negative/adverse articles only
- Summary: Analysis statistics and metadata

Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Corporate Intelligence API
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach Excel file
            with open(excel_file_path, 'rb') as f:
                excel_attachment = MIMEApplication(f.read())
                excel_attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=excel_path.name
                )
                msg.attach(excel_attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Excel results successfully sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_search_results_excel(self, excel_file_path: str, company_name: str) -> bool:
        """
        Send search results Excel file via email with customized message.
         
        Args:
            excel_file_path: Path to the search results Excel file
            company_name: Name of the company searched
            
        Returns:
            Boolean indicating success/failure
        """
        try:
            # Verify Excel file exists
            excel_path = Path(excel_file_path)
            if not excel_path.exists():
                logger.error(f"Search results Excel file not found: {excel_file_path}")
                return False
            
            # Get file info
            file_stat = excel_path.stat()
            file_size_mb = round(file_stat.st_size / (1024 * 1024), 2)
            file_date = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Search Results - {company_name}"
            
            # Email body for search results
            body = f"""
Hello,

Please find attached the search results for {company_name}.

File Details:
- Company: {company_name}
- File: {excel_path.name}
- Size: {file_size_mb} MB
- Generated: {file_date}

The Excel file contains:
- Search queries executed
- Article URLs and metadata
- Search engine sources (Google, DuckDuckGo)
- Deduplication statistics
- Time-filtered results
- Comprehensive search coverage

This file contains raw search results that can be further processed for detailed analysis.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Corporate Intelligence Search API
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach Excel file
            with open(excel_file_path, 'rb') as f:
                excel_attachment = MIMEApplication(f.read())
                excel_attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=excel_path.name
                )
                msg.attach(excel_attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Search results Excel file successfully sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send search results email: {str(e)}")
            return False

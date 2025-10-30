"""
AWS S3 Service for uploading Excel files and generating download URLs
"""
import boto3
import os
import logging
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Service:
    """Service for uploading files to AWS S3 and generating download URLs"""
    
    def __init__(self, bucket_name: Optional[str] = None, region_name: str = 'us-east-1'):
        """
        Initialize S3 service
        
        Args:
            bucket_name: S3 bucket name (will use env var AWS_S3_BUCKET if not provided)
            region_name: AWS region name (default: us-east-1)
        """
        self.bucket_name = bucket_name or os.getenv('AWS_S3_BUCKET')
        self.region_name = region_name
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided either as parameter or AWS_S3_BUCKET environment variable")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=self.region_name)
            logger.info(f"✅ S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {str(e)}")
            raise
    
    def upload_file(self, file_path: str, s3_key: Optional[str] = None, 
                   content_type: str = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') -> Dict[str, Any]:
        """
        Upload a file to S3
        
        Args:
            file_path: Local path to the file to upload
            s3_key: S3 object key (if None, uses filename from file_path)
            content_type: MIME type for the file
            
        Returns:
            Dict containing upload result with keys: success, s3_key, download_url, error
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
                "s3_key": None,
                "download_url": None
            }
        
        # Generate S3 key if not provided
        if not s3_key:
            filename = os.path.basename(file_path)
            # Add folder structure for better organization
            s3_key = f"search-results/{filename}"
        
        try:
            logger.info(f"📤 Uploading {file_path} to s3://{self.bucket_name}/{s3_key}")
            
            # Upload file with proper metadata
            extra_args = {
                'ContentType': content_type,
                'ContentDisposition': f'attachment; filename="{os.path.basename(file_path)}"',
                'Metadata': {
                    'uploaded-by': 'article-analysis-api',
                    'upload-timestamp': str(int(__import__('time').time()))
                }
            }
            
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs=extra_args
            )
            
            # Generate download URL
            download_url = self._generate_download_url(s3_key)
            
            logger.info(f"✅ File uploaded successfully to S3: {s3_key}")
            
            return {
                "success": True,
                "s3_key": s3_key,
                "download_url": download_url,
                "bucket_name": self.bucket_name,
                "error": None
            }
            
        except NoCredentialsError:
            error_msg = "AWS credentials not found. Please configure AWS credentials."
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key,
                "download_url": None
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"AWS S3 error ({error_code}): {e.response['Error']['Message']}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key,
                "download_url": None
            }
            
        except Exception as e:
            error_msg = f"Unexpected error uploading to S3: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "s3_key": s3_key,
                "download_url": None
            }
    
    def _generate_download_url(self, s3_key: str) -> str:
        """
        Generate a direct public S3 URL for downloading files
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Direct public S3 URL
        """
        try:
            # Generate direct public S3 URL (bucket must be public)
            public_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Generated public S3 URL: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Error generating public URL: {str(e)}")
            raise
    
    
    def check_bucket_exists(self) -> bool:
        """
        Check if the configured S3 bucket exists and is accessible
        
        Returns:
            True if bucket exists and is accessible, False otherwise
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ S3 bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"❌ S3 bucket '{self.bucket_name}' does not exist")
            else:
                logger.error(f"❌ Error accessing S3 bucket '{self.bucket_name}': {error_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error checking S3 bucket: {str(e)}")
            return False
    
    def delete_file(self, s3_key: str) -> Dict[str, Any]:
        """
        Delete a file from S3 (optional cleanup method)
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            Dict containing deletion result
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"🗑️ File deleted from S3: {s3_key}")
            return {"success": True, "error": None}
        except Exception as e:
            error_msg = f"Error deleting file from S3: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}



# Convenience function for quick uploads
def upload_excel_to_s3(file_path: str, bucket_name: Optional[str] = None, 
                      custom_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to upload an Excel file to S3
    
    Args:
        file_path: Path to Excel file
        bucket_name: S3 bucket name (uses env var if not provided)
        custom_key: Custom S3 key (uses filename if not provided)
        
    Returns:
        Upload result dictionary
    """
    try:
        s3_service = S3Service(bucket_name=bucket_name)
        result = s3_service.upload_file(
            file_path=file_path,
            s3_key=custom_key,
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize S3 service: {str(e)}",
            "s3_key": None,
            "download_url": None
        }

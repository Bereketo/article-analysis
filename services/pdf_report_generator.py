"""
PDF Report Generator for Adverse Media Analysis
===============================================

Generates professional PDF reports with formatted tables showing adverse press and media findings.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import os
from datetime import datetime
from typing import List, Dict, Any
import textwrap
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()


class AdverseMediaPDFReport:
    """Generate PDF reports for adverse media analysis"""
    
    def __init__(self):
        # Color scheme matching the image
        self.header_blue = HexColor('#4472C4')  # Dark blue header
        self.subheader_blue = HexColor('#5B9BD5')  # Light blue subheader
        self.high_risk_color = colors.red
        self.medium_risk_color = colors.orange
        self.low_risk_color = colors.green
        
        # Initialize LLM for enhanced key findings generation
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY')
        )
        
        # Styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.white,
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
    async def create_adverse_media_table(self, adverse_findings: List[Dict[str, Any]], company_name: str) -> Table:
        """Create the main adverse media findings table"""
        
        # Table headers
        headers = [
            ['Adverse press and media', '', '', ''],
            ['Key findings', 'Source', 'Report date', 'Source reliability']
        ]
        
        # Add overall summary row first
        if adverse_findings:
            overall_summary = await self._create_detailed_company_summary(adverse_findings, company_name)
            summary_para = Paragraph(overall_summary, self.styles['BodyText'])
            data_rows = [[
                summary_para,
                "-",  # No source for summary
                "-",  # No date for summary  
                "-"   # No reliability for summary
            ]]
        else:
            data_rows = []
        
        # Process findings into table rows
        for finding in adverse_findings:
            # Clean and format key findings text
            key_finding_text = finding.get('key_finding', '')
            
            # Clean up text before wrapping
            cleaned_text = self._clean_text(key_finding_text)
            
            # Create paragraph without pre-wrapping (let ReportLab handle line breaks)
            key_finding_para = Paragraph(cleaned_text, self.styles['BodyText'])
            
            # Format date
            report_date = finding.get('report_date', '')
            if isinstance(report_date, datetime):
                report_date = report_date.strftime('%d-%b-%y')
            
            # Handle source (single or multiple)
            source_text = finding.get('source', '')
            source_url = finding.get('source_url', '')
            
            if '<br/><br/>' in source_text:  # Pre-formatted HTML sources (grouped)
                # Sources are already formatted with proper links
                source_para = Paragraph(source_text, self.styles['BodyText'])
            elif '\n' in source_text:  # Multiple sources (grouped) - fallback
                sources = source_text.split('\n')
                source_lines = []
                for source in sources:
                    if source and source.strip():
                        source_lines.append(f'<font color="blue"><u>{source.strip()}</u></font>')
                source_para = Paragraph('<br/><br/>'.join(source_lines), self.styles['BodyText'])
            elif source_url and source_url.strip():
                source_para = Paragraph(f'<link href="{source_url}" color="blue"><u>{source_text}</u></link>', 
                                       self.styles['BodyText'])
            else:
                source_para = Paragraph(f'<font color="blue"><u>{source_text}</u></font>', self.styles['BodyText'])
            
            # Handle report date (single or multiple) with consistent formatting
            report_date = finding.get('report_date', '')
            if '\n' in report_date:  # Multiple dates (grouped)
                dates = report_date.split('\n')
                date_lines = '<br/><br/>'.join(dates)  # Add proper spacing between dates
                date_para = Paragraph(date_lines, self.styles['BodyText'])
            else:
                if isinstance(report_date, datetime):
                    report_date = report_date.strftime('%d-%b-%y')
                # Wrap single date in Paragraph for consistent formatting
                date_para = Paragraph(report_date, self.styles['BodyText'])
            
            # Handle reliability (single or multiple)
            reliability = finding.get('reliability', 'Medium')
            if '\n' in reliability:  # Multiple reliabilities (grouped)
                reliabilities = reliability.split('\n')
                reliability_lines = []
                for rel in reliabilities:
                    if rel.lower() == 'high':
                        reliability_lines.append(f'<font color="red"><b>{rel}</b></font>')
                    elif rel.lower() == 'medium':
                        reliability_lines.append(f'<font color="orange"><b>{rel}</b></font>')
                    else:
                        reliability_lines.append(f'<font color="green"><b>{rel}</b></font>')
                reliability_para = Paragraph('<br/><br/>'.join(reliability_lines), self.styles['BodyText'])  # Add spacing
            else:
                reliability_para = Paragraph(reliability, self._get_reliability_style(reliability))
            
            data_rows.append([
                key_finding_para,
                source_para,
                date_para,  # Always use date_para for consistent formatting
                reliability_para
            ])
        
        # If no findings, add a "No adverse findings" row
        if not data_rows:
            data_rows.append([
                Paragraph("No adverse media findings identified", self.styles['BodyText']),
                "-",
                "-", 
                "-"
            ])
        
        # Combine headers and data
        table_data = headers + data_rows
        
        # Create table with column widths - optimized for grouped content
        col_widths = [5.5*inch, 1.2*inch, 0.9*inch, 1.1*inch]  # Wider key findings column
        table = Table(table_data, colWidths=col_widths, repeatRows=0, splitByRow=True)  # No header repetition
        
        # Apply table styling with better handling for large content and summary row
        table_style = TableStyle([
            # Header row 1 styling (dark blue)
            ('BACKGROUND', (0, 0), (-1, 0), self.header_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('SPAN', (0, 0), (-1, 0)),  # Merge first row cells
            
            # Header row 2 styling (light blue)
            ('BACKGROUND', (0, 1), (-1, 1), self.subheader_blue),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('ALIGN', (0, 1), (-1, 1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 10),
            
            # Summary row styling (first data row) - highlighted
            ('BACKGROUND', (0, 2), (-1, 2), HexColor('#E8F4FD')),  # Light blue background
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 2), (-1, 2), 9),
            ('ALIGN', (0, 2), (-1, 2), 'LEFT'),
            ('VALIGN', (0, 2), (-1, 2), 'TOP'),
            
            # Regular data rows styling (starting from row 3)
            ('FONTNAME', (0, 3), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 3), (-1, -1), 8),
            ('ALIGN', (0, 3), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 3), (-1, -1), 'TOP'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('LINEBELOW', (0, 1), (-1, 1), 1, colors.black),
            ('LINEBELOW', (0, 2), (-1, 2), 1, colors.grey),  # Separate summary row
            
            # Alternating row colors for data rows (excluding summary)
            ('ROWBACKGROUNDS', (0, 3), (-1, -1), [colors.white, HexColor('#F8F9FA')]),
            
            # Reduced padding for more space
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ])
        
        table.setStyle(table_style)
        return table
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text to remove unnecessary gaps and formatting issues"""
        if not text:
            return text
        
        # Check if this is formatted grouped content (contains HTML tags)
        if '<br/>' in text or '<font' in text or '&nbsp;' in text:
            # For grouped content, preserve HTML formatting but clean up spacing
            cleaned = text
            # Only normalize excessive spaces but keep HTML structure
            while '  ' in cleaned:
                cleaned = cleaned.replace('  ', ' ')
            return cleaned.strip()
        else:
            # For regular content, do normal cleaning
            # Remove excessive whitespace, newlines, and normalize spacing
            cleaned = ' '.join(text.split())
            
            # Remove common formatting artifacts
            cleaned = cleaned.replace('\t', ' ')  # Replace tabs with spaces
            cleaned = cleaned.replace('\n', ' ')  # Replace newlines with spaces
            cleaned = cleaned.replace('\r', ' ')  # Replace carriage returns
            
            # Fix common spacing issues around punctuation
            cleaned = cleaned.replace(' .', '.')  # Remove space before period
            cleaned = cleaned.replace(' ,', ',')  # Remove space before comma
            cleaned = cleaned.replace(' ;', ';')  # Remove space before semicolon
            cleaned = cleaned.replace(' :', ':')  # Remove space before colon
            
            # Normalize multiple spaces to single space
            while '  ' in cleaned:
                cleaned = cleaned.replace('  ', ' ')
            
            return cleaned.strip()
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width"""
        return '<br/>'.join(textwrap.wrap(text, width))
    
    def _get_reliability_style(self, reliability: str) -> ParagraphStyle:
        """Get paragraph style based on reliability level"""
        base_style = self.styles['BodyText']
        
        if reliability.lower() == 'high':
            color = self.high_risk_color
        elif reliability.lower() == 'medium':
            color = self.medium_risk_color
        else:
            color = self.low_risk_color
            
        return ParagraphStyle(
            f'Reliability_{reliability}',
            parent=base_style,
            textColor=color,
            fontName='Helvetica-Bold'
        )
    
    async def generate_report(
        self,
        company_name: str,
        adverse_findings: List[Dict[str, Any]],
        output_path: str,
        report_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Generate complete PDF report
        
        Args:
            company_name: Name of the company analyzed
            adverse_findings: List of adverse media findings
            output_path: Path to save the PDF
            report_metadata: Additional metadata for the report
        
        Returns:
            Path to the generated PDF file
        """
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(letter),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        
        # Build content
        story = []
        
        # Title
        title = Paragraph(f"Adverse Media Analysis Report", self.styles['Title'])
        story.append(title)
        
        # Company name
        company_para = Paragraph(f"<b>Company:</b> {company_name}", self.styles['Heading2'])
        story.append(company_para)
        
        # Report date
        report_date = datetime.now().strftime("%d %B %Y")
        date_para = Paragraph(f"<b>Report Date:</b> {report_date}", self.styles['Normal'])
        story.append(date_para)
        
        # Time period if available
        if report_metadata and report_metadata.get('time_period'):
            time_period = report_metadata['time_period']
            period_para = Paragraph(f"<b>Time Period Analyzed:</b> {time_period}", self.styles['Normal'])
            story.append(period_para)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Skip metadata summary - removed per user request
        
        # Main findings table
        findings_table = await self.create_adverse_media_table(adverse_findings, company_name)
        story.append(findings_table)
        
        # Add footer space and source reliability explanation
        story.append(Spacer(1, 0.4*inch))
        
        # Footer note about source reliability
        footer_style = ParagraphStyle(
            'Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_LEFT
        )
        
        reliability_note = Paragraph(
            "<b>Source Reliability:</b> High (Tier-1 financial/news sources, confidence >80%), Medium (Regional sources, confidence 50-80%), Low (Other sources, confidence <50%).<br/>"
            "Source reliability is determined based on publication credibility and AI confidence scores in risk assessment.",
            footer_style
        )
        story.append(reliability_note)
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    async def generate_from_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str = "pdf-reports",
        time_period: str = None
    ) -> str:
        """
        Generate PDF from analysis results (integration with existing pipeline)
        
        Args:
            analysis_results: Results from article analysis endpoint
            output_dir: Directory to save PDF reports
            time_period: Time period for the analysis (e.g., "Last 12 months", "2024-01-01 to 2024-12-31")
        
        Returns:
            Path to generated PDF
        """
        
        # Extract company name and findings from results
        company_name = analysis_results.get('company_name', 'Unknown Company')
        
        # Convert analysis results to adverse findings format
        adverse_articles = []
        
        results = analysis_results.get('results', [])
        for result in results:
            # Extract is_adverse from the correct nested structure
            analysis = result.get('analysis', {})
            is_adverse = analysis.get('is_adverse', 'Neutral')
            
            # Only include adverse/negative findings
            if is_adverse == 'Negative':
                # Check if article is stock-related and should be excluded
                if await self._is_stock_related_article(result):
                    print(f"💹 Skipping stock-related article: {result.get('url', 'No URL')}")
                    continue
                
                # Double-check if article is actually adverse content
                if not await self._is_actually_adverse_content(result):
                    print(f"⚠️  Skipping non-adverse content: {result.get('url', 'No URL')}")
                    continue
                    
                print(f"🔍 Processing adverse article: {result.get('url', 'No URL')}")
                adverse_articles.append(result)
            else:
                print(f"⏭️  Skipping non-adverse article: {result.get('url', 'No URL')} (is_adverse: {is_adverse})")
        
        # Group related articles semantically
        print(f"🔄 Starting grouping for {len(adverse_articles)} adverse articles")
        article_groups = await self._group_articles_semantically(adverse_articles, company_name)
        print(f"📊 Grouping result: {len(article_groups)} groups created")
        
        # Convert groups to adverse findings format
        adverse_findings = []
        for i, group in enumerate(article_groups):
            print(f"📋 Group {i+1}: {group['theme']} with {len(group['articles'])} articles")
            if len(group['articles']) == 1:
                # Single article - process normally
                print(f"   → Processing as single article")
                result = group['articles'][0]
                source_name = self._extract_source_name(result)
                analysis = result.get('analysis', {})
                
                finding = {
                    'key_finding': await self._generate_enhanced_key_finding(result),
                    'source': source_name,
                    'source_url': result.get('url', ''),
                    'report_date': self._parse_date(analysis.get('published_date', '')),
                    'reliability': self._determine_reliability(result),
                    'raw_date': analysis.get('published_date', ''),
                }
                adverse_findings.append(finding)
            else:
                # Grouped articles - create grouped finding
                print(f"   → Processing as grouped finding")
                grouped_finding = await self._create_grouped_finding(group)
                adverse_findings.append(grouped_finding)
        
        # Sort adverse findings by credibility and relevance, then by date (latest first)
        # This ensures groups with the most recent articles appear first in the report
        adverse_findings = self._sort_findings_by_credibility_and_date(adverse_findings)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_path = os.path.join(output_dir, f"{safe_company_name}_adverse_media_{timestamp}.pdf")
        
        # Generate metadata with time period
        metadata = {
            'total_articles_analyzed': len(results),
            'adverse_articles_found': len(adverse_findings),
            'analysis_timestamp': analysis_results.get('processed_at', datetime.now().isoformat()),
            'time_period': time_period or self._determine_time_period(adverse_findings)
        }
        
        # Generate report
        return await self.generate_report(company_name, adverse_findings, output_path, metadata)
    
    async def _generate_enhanced_key_finding(self, result: Dict[str, Any]) -> str:
        """Generate comprehensive, professional key finding using LLM"""
        
        try:
            # Extract all available information
            title = result.get('title', 'Untitled Article')
            content = result.get('content', '')
            analysis = result.get('analysis', {})
            metadata = analysis.get('metadata', {})
            source = result.get('source', 'Unknown Source')
            
            # Get existing analysis details if available
            risk_explanation = (
                metadata.get('risk_explanation') or 
                metadata.get('risk_snippet') or
                analysis.get('is_adverse_reason') or
                analysis.get('risk_reason') or
                ''
            )
            
            summary = analysis.get('summary') or metadata.get('summary') or ''
            
            # Get publication date for the "On [date]" format
            published_date = analysis.get('published_date', '')
            
            # Create focused input for LLM
            input_data = f"""
ARTICLE TITLE: {title}

PUBLISHED DATE: {published_date}

ARTICLE CONTENT: {content[:800]}{'...' if len(content) > 800 else ''}

EXISTING ANALYSIS: {risk_explanation}

SUMMARY: {summary}

SOURCE: {source}
"""
            
            system_prompt = """You are a professional financial risk analyst writing key findings for adverse media reports. 

Your task is to create a concise, professional key finding paragraph that:

1. STARTS with "On [date]," using the publication date from the article
2. Provides ESSENTIAL context and key details only
3. Explains the SPECIFIC adverse event clearly
4. Includes CONCRETE details like amounts, parties involved
5. Maintains an OBJECTIVE, factual tone
6. Is CONCISE (aim for 80-150 words maximum - just one paragraph)
7. Focuses on the MAIN adverse event


Format: "On [DD Month, YYYY], [key adverse event details]. [Brief impact/consequence if relevant]."

Example style: "On 15 January, 2024, the Securities and Exchange Board of India imposed penalties of INR 25 lakh on the company for violations related to disclosure norms. The regulatory action stemmed from inadequate reporting of related party transactions during the previous fiscal year."

Be factual, concise, and professional. Keep it to one focused paragraph only."""
            
            human_prompt = f"""Based on the following article information, create a concise key finding paragraph for an adverse media report:

{input_data}

Generate a SHORT, professional key finding that:
- Starts with "On [date]" using the published date
- Captures the main adverse event in 80-150 words maximum
- Focuses only on the key facts and implications
- Is written as one focused paragraph"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            enhanced_finding = response.content.strip()
            
            # Clean up and ensure reasonable length
            enhanced_finding = ' '.join(enhanced_finding.split())
            
            # Limit length for concise findings but don't cut mid-sentence
            if len(enhanced_finding) > 800:
                # Find the last complete sentence within the limit
                truncated = enhanced_finding[:797]
                last_period = truncated.rfind('.')
                if last_period > 400:  # Only if we have a reasonable amount of content
                    enhanced_finding = enhanced_finding[:last_period + 1]
                else:
                    enhanced_finding = truncated + '...'
            
            print(f"🤖 Generated enhanced key finding: {enhanced_finding[:200]}...")
            return enhanced_finding
            
        except Exception as e:
            print(f"❌ Error generating enhanced key finding: {e}")
            # Fallback to simpler extraction
            return self._extract_simple_key_finding(result)
    
    def _extract_simple_key_finding(self, result: Dict[str, Any]) -> str:
        """Fallback method for key finding extraction if LLM fails"""
        
        # Extract basic information
        title = result.get('title', '')
        content = result.get('content', '')
        analysis = result.get('analysis', {})
        
        # Start with title
        parts = []
        if title:
            parts.append(f"<b>{title}</b>")
        
        # Add substantial content for detail
        if content:
            content_snippet = content[:1200] + '...' if len(content) > 1200 else content
            content_snippet = ' '.join(content_snippet.split())
            parts.append(content_snippet)
        
        # Add any existing analysis
        summary = analysis.get('summary', '')
        if summary:
            parts.append(summary)
        
        if parts:
            key_finding = ' '.join(parts)
            key_finding = ' '.join(key_finding.split())
            return key_finding[:1800] + '...' if len(key_finding) > 1800 else key_finding
        
        return "Adverse media content identified requiring further analysis."
    
    async def _is_stock_related_article(self, result: Dict[str, Any]) -> bool:
        """Use LLM to determine if article is stock/trading related and should be excluded"""
        
        title = result.get('title', '')
        content = result.get('content', '')[:600]  # Limit content for efficiency
        
        # Quick keyword check first for obvious cases to save API calls
        text_to_check = f"{title} {content}".lower()
        obvious_stock_terms = [
            'stock price', 'share price', 'trading session', 'market close', 'closing price',
            'opening price', 'stock surge', 'shares plummet', 'price target', 'analyst rating'
        ]
        if any(term in text_to_check for term in obvious_stock_terms):
            print(f"💹 Quick-filtered stock article: {title[:50]}...")
            return True
        
        system_prompt = """You are analyzing news articles to identify stock trading and price movement content that should be excluded from adverse media reports.

Return ONLY "YES" or "NO" to indicate if the article is primarily about:
- Stock prices, share prices, or market valuations
- Trading activities, market movements, or investment recommendations  
- Stock market performance, price targets, or analyst ratings
- Market capitalization changes or shareholder value impacts from price movements
- Investment advice, buy/sell recommendations, or portfolio discussions

Return "NO" if the article is about:
- Regulatory violations, compliance issues, legal matters
- Business operations, corporate governance, management changes
- Financial results, earnings, or business performance (unless focused on stock reaction)
- Investigations, penalties, fraud, or adverse business events
- Operational issues, product problems, or service disruptions

Focus on the PRIMARY purpose and main content of the article, not incidental stock mentions."""
        
        human_prompt = f"""Title: {title}

Content: {content}

Is this article primarily about stock trading, prices, or market movements?"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            result_text = response.content.strip().upper()
            
            is_stock_related = result_text == "YES"
            if is_stock_related:
                print(f"💹 LLM identified stock article: {title[:50]}...")
            return is_stock_related
            
        except Exception as e:
            print(f"❌ Error in LLM stock filtering: {e}")
            # Fallback to conservative approach - don't filter if uncertain
            return False
    
    async def _is_actually_adverse_content(self, result: Dict[str, Any]) -> bool:
        """Validate if article contains actual adverse content (not support/positive news)"""
        
        title = result.get('title', '')
        content = result.get('content', '')[:800]  # More content for better analysis
        
        system_prompt = """You are validating if articles contain ACTUAL adverse/negative content for risk reporting.

Return "YES" if the article contains:
- Regulatory violations, penalties, fines, investigations
- Legal issues, compliance failures, governance problems  
- Fraud allegations, financial misconduct, operational failures
- Criminal charges, lawsuits, regulatory sanctions
- Show cause notices, enforcement actions

Return "NO" if the article contains:
- Support announcements, backing declarations, confidence statements
- Investment commitments, partnership announcements
- Business growth, expansions, market opportunities
- Positive reviews of ties, reaffirmation of support
- General business news without adverse events

Focus on the PRIMARY content and intent of the article."""
        
        human_prompt = f"""Title: {title}

Content: {content}

Does this article contain actual adverse/negative content that should be included in a risk report?"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            result_text = response.content.strip().upper()
            
            is_adverse = result_text == "YES"
            if not is_adverse:
                print(f"✋ LLM identified non-adverse content: {title[:50]}...")
            return is_adverse
            
        except Exception as e:
            print(f"❌ Error in adverse content validation: {e}")
            # Fallback: if uncertain, include it (conservative approach)
            return True
    
    def _create_overall_summary(self, adverse_findings: List[Dict[str, Any]]) -> str:
        """Create brief overall summary of all adverse findings"""
        
        total_findings = len(adverse_findings)
        
        # Count grouped vs individual findings
        grouped_findings = sum(1 for f in adverse_findings if '\n' in f.get('source', ''))
        individual_findings = total_findings - grouped_findings
        
        # Create summary based on content analysis
        if total_findings == 0:
            return "No adverse media findings identified through comprehensive analysis."
        elif total_findings == 1:
            return "One adverse media finding has been identified regarding the company's operations and regulatory compliance."
        else:
            summary_parts = []
            
            if total_findings <= 5:
                summary_parts.append(f"Multiple adverse media findings ({total_findings} items) have been identified")
            else:
                summary_parts.append(f"Significant adverse media findings ({total_findings} items) have been identified")
            
            # Add context about types of issues
            summary_parts.append("relating to regulatory compliance, operational challenges, and corporate governance matters.")
            
            if grouped_findings > 0:
                summary_parts.append(f"The findings include {grouped_findings} thematic groups of related incidents and {individual_findings} individual matters.")
            
            summary_parts.append("These findings require careful consideration in risk assessment and due diligence processes.")
            
            return " ".join(summary_parts)
    
    async def _create_detailed_company_summary(self, adverse_findings: List[Dict[str, Any]], company_name: str) -> str:
        """Create detailed company-specific summary like the NIACL example"""
        
        if not adverse_findings:
            return f"{company_name} has been analyzed for adverse media coverage with no significant findings identified."
        
        # Extract key information from findings
        finding_details = []
        for finding in adverse_findings:
            key_finding = finding.get('key_finding', '')
            # Extract first 200 characters of each finding for analysis
            if key_finding:
                detail = key_finding[:200].replace('<br/>', ' ').replace('<font color=', '').replace('</font>', '')
                detail = ' '.join(detail.split())  # Clean up spacing
                finding_details.append(detail)
        
        findings_text = '\n'.join(finding_details[:10])  # Limit to first 10 for analysis
        
        system_prompt = f"""You are writing an executive summary for an adverse media report about {company_name}. 

Create a professional summary that follows this structure:
1. Start with "{company_name} has a high public profile. Online references to the company are primarily in relation to its business activities."
2. Then state: "{company_name} has been subject of adverse press in relation to [specific issues found]"
3. Provide specific details about penalties, regulatory actions, amounts, time periods, and regulatory bodies
4. Include context about materiality when relevant
5. List specific adverse events like penalties, investigations, regulatory actions
6. End with "among other issues" or similar

Style: Professional, factual, comprehensive. Similar to the NIACL example format.
Length: 150-300 words in paragraph form.
Currency: Use INR, USD format (not ₹, $).
Include specific amounts, dates, regulatory bodies when mentioned."""
        
        human_prompt = f"""Create an executive summary for {company_name} based on these adverse findings:

{findings_text}

Write a comprehensive summary following the format and style requirements."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            summary = response.content.strip()
            
            # Ensure reasonable length but don't cut mid-sentence
            if len(summary) > 1000:
                # Find the last complete sentence within reasonable length
                truncated = summary[:997]
                last_period = truncated.rfind('.')
                last_semicolon = truncated.rfind(';')
                
                # Use the latest sentence ending
                last_sentence_end = max(last_period, last_semicolon)
                
                if last_sentence_end > 500:  # Only if we have substantial content
                    summary = summary[:last_sentence_end + 1]
                else:
                    # If no good sentence break, try to end at a reasonable word boundary
                    words = summary[:900].split()
                    summary = ' '.join(words) + '...'
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating detailed summary: {e}")
            # Fallback to basic summary
            return f"{company_name} has a high public profile. Online references to the company are primarily in relation to its business activities. {company_name} has been subject of adverse press in relation to various regulatory and operational matters identified through comprehensive media analysis."
    
    async def _group_articles_semantically(self, articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
        """Group related articles using LLM semantic analysis with two-step duplicate detection"""
        
        print(f"🤖 Semantic grouping: {len(articles)} articles for {company_name}")
        if len(articles) <= 1:
            print(f"   → Too few articles ({len(articles)}), returning as individual")
            return [{'theme': 'Individual', 'articles': articles}]
        
        # STEP 1: First detect duplicates
        print(f"   🔍 Step 1: Detecting duplicates...")
        duplicate_indices = await self._detect_duplicates(articles, company_name)
        
        # Remove duplicates from articles list
        filtered_articles = []
        for i, article in enumerate(articles):
            if i not in duplicate_indices:
                filtered_articles.append(article)
            else:
                print(f"   ❌ Excluding duplicate: {article.get('title', 'No title')[:60]}...")
        
        print(f"   📊 After duplicate removal: {len(filtered_articles)} articles remaining")
        
        if len(filtered_articles) <= 1:
            print(f"   → Too few articles after deduplication, returning as individual")
            return [{'theme': 'Individual', 'articles': filtered_articles}]
        
        # STEP 2: Date-based deduplication (same/close dates for same company)
        print(f"   📅 Step 2: Date-based deduplication for same-company events...")
        date_deduplicated_articles = self._date_based_deduplication(filtered_articles)
        
        if len(date_deduplicated_articles) <= 1:
            print(f"   → Too few articles after date deduplication, returning as individual")
            return [{'theme': 'Individual', 'articles': date_deduplicated_articles}]
        
        # STEP 3: Group remaining unique articles
        print(f"   📁 Step 3: Grouping {len(date_deduplicated_articles)} unique articles...")
        groups = await self._group_unique_articles(date_deduplicated_articles, company_name)
        
        # Note: No need for post-processing same-date duplicates since we already did date deduplication
        return groups
    
    async def _detect_duplicates(self, articles: List[Dict[str, Any]], company_name: str) -> set:
        """Detect duplicate articles using LLM - returns indices of duplicates to remove"""
        
        # Prepare article summaries for duplicate detection
        article_summaries = []
        for i, article in enumerate(articles):
            title = article.get('title', f'Article {i+1}')
            content = article.get('content', '')[:300]  # More content for better duplicate detection
            date = article.get('analysis', {}).get('published_date', 'No date')
            article_summaries.append(f"Article {i+1}: [{date}] {title} - {content}...")
            print(f"   📄 Article {i+1}: {title[:50]}...")
        
        duplicate_prompt = f"""You are a duplicate detector for news articles about {company_name}.

Your ONLY task is to identify articles reporting the EXACT SAME EVENT.

DUPLICATE CRITERIA - Articles are duplicates if they have:
• Same date + same company + same regulatory action
• Same penalty amount + same company + same regulator
• Same incident + same people affected + same company
• Identical or nearly identical titles

EXAMPLES:
❌ "Dec 3, 2024 Labour Ministry show cause notice Adani Ports ₹5cr 2500 workers"
❌ "3 December 2024 Adani Ports Labour Ministry notice ₹5 crore 2500 workers"
→ DUPLICATES! Same date, company, amount, workers

❌ "SEBI fines Adani Group Rs 25 crore disclosure violations"
❌ "SEBI fines Adani Group Rs 25 crore disclosure violations" 
→ DUPLICATES! Identical titles

Return ONLY a JSON array of article numbers to REMOVE (keep the earliest, remove later ones):
{{
  "duplicate_articles": [4, 6, 7]
}}

Scan each article pair carefully. Mark later-numbered articles as duplicates.
Return ONLY JSON, no other text."""
        
        articles_text = "\n".join(article_summaries)
        human_prompt = f"Detect duplicates in these {company_name} articles:\n{articles_text}"
        
        try:
            messages = [
                SystemMessage(content=duplicate_prompt),
                HumanMessage(content=human_prompt)
            ]
            print(f"   💬 Sending duplicate detection request to LLM...")
            response = await self.llm.ainvoke(messages)
            
            raw_content = response.content.strip()
            print(f"   ✅ Duplicate detection response: {raw_content[:100]}...")
            
            # Clean up response
            if raw_content.startswith('```json'):
                raw_content = raw_content.replace('```json', '').replace('```', '').strip()
            
            duplicate_result = json.loads(raw_content)
            duplicate_articles = duplicate_result.get('duplicate_articles', [])
            
            print(f"   🔍 Found {len(duplicate_articles)} duplicates: {duplicate_articles}")
            
            # Convert to 0-based indices and return as set
            duplicate_indices = set(i-1 for i in duplicate_articles if 0 < i <= len(articles))
            return duplicate_indices
            
        except Exception as e:
            print(f"   ⚠️ Duplicate detection failed: {e}")
            return set()  # Return empty set if detection fails
    
    def _date_based_deduplication(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple date-based deduplication - keep first article from each date"""
        
        if len(articles) <= 1:
            return articles
            
        seen_dates = set()
        unique_articles = []
        total_removed = 0
        
        for article in articles:
            date_str = article.get('analysis', {}).get('published_date', '')
            # Normalize date (remove time, keep just YYYY-MM-DD)
            normalized_date = date_str.split('T')[0] if date_str else 'no_date'
            
            if normalized_date not in seen_dates:
                # First article with this date - keep it
                seen_dates.add(normalized_date)
                unique_articles.append(article)
            else:
                # Already have article with this date - skip it
                total_removed += 1
                print(f"      ❌ Removed duplicate for {normalized_date}: {article.get('title', 'No title')[:60]}...")
        
        if total_removed > 0:
            print(f"   ✅ Date deduplication: Removed {total_removed} articles, kept {len(unique_articles)}")
        else:
            print(f"   ✅ Date deduplication: No date-based duplicates found")
        
        return unique_articles
    
    def _remove_same_date_duplicates_from_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process groups to remove same-date duplicates within each group"""
        
        cleaned_groups = []
        total_removed = 0
        
        for group in groups:
            articles = group['articles']
            theme = group['theme']
            
            if len(articles) <= 1:
                # Single article groups don't need cleaning
                cleaned_groups.append(group)
                continue
                
            # Group articles by date
            date_groups = {}
            for article in articles:
                date = article.get('analysis', {}).get('published_date', 'Unknown')
                # Normalize date format for comparison
                normalized_date = self._normalize_date(date)
                
                if normalized_date not in date_groups:
                    date_groups[normalized_date] = []
                date_groups[normalized_date].append(article)
            
            # Keep only the first article from each date group (preferring earlier URLs/sources)
            cleaned_articles = []
            for date, date_articles in date_groups.items():
                if len(date_articles) == 1:
                    # Single article for this date, keep it
                    cleaned_articles.extend(date_articles)
                else:
                    # Multiple articles for same date, keep the best one
                    kept_article = self._select_best_article_from_date_group(date_articles)
                    cleaned_articles.append(kept_article)
                    removed_count = len(date_articles) - 1
                    total_removed += removed_count
                    
                    print(f"      ⚠️  Removed {removed_count} same-date duplicate(s) for {date} in group '{theme}'")
                    for i, removed_article in enumerate(date_articles):
                        if removed_article != kept_article:
                            print(f"         - Removed: {removed_article.get('title', 'No title')[:60]}...")
            
            # Update group with cleaned articles
            if cleaned_articles:
                cleaned_group = {
                    'theme': theme,
                    'articles': cleaned_articles
                }
                cleaned_groups.append(cleaned_group)
            # If no articles remain after cleaning, skip the group entirely
        
        if total_removed > 0:
            print(f"   ✅ Post-processing complete: Removed {total_removed} same-date duplicates from groups")
        else:
            print(f"   ✅ Post-processing complete: No same-date duplicates found")
        
        return cleaned_groups
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string for comparison"""
        if not date_str or date_str == 'Unknown':
            return 'Unknown'
        
        # Remove time components and normalize format
        date_only = date_str.split('T')[0].split(' ')[0]
        return date_only.strip()
    
    def _select_best_article_from_date_group(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best article from a group of same-date articles"""
        
        # Preference order: 
        # 1. Longest content (more comprehensive)
        # 2. Higher reliability sources
        # 3. Earlier in original list (alphabetical URL order)
        
        def article_score(article):
            content_length = len(article.get('content', ''))
            source = article.get('source', '').lower()
            
            # Reliability bonus based on source
            reliability_bonus = 0
            high_reliability_sources = ['reuters', 'bloomberg', 'economic times', 'business standard', 'mint']
            for reliable_source in high_reliability_sources:
                if reliable_source in source:
                    reliability_bonus = 1000
                    break
            
            return content_length + reliability_bonus
        
        # Select article with highest score
        best_article = max(articles, key=article_score)
        return best_article
    
    async def _group_unique_articles(self, articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
        """Group unique articles (after duplicates removed) by themes with robust error handling"""
        
        if not articles:
            print(f"   ⚠️ No articles to group")
            return []
        
        if len(articles) == 1:
            print(f"   🔍 Single article remaining, returning as individual")
            return [{'theme': 'Individual', 'articles': articles}]
        
        # Prepare article summaries for grouping with more context
        article_summaries = []
        for i, article in enumerate(articles):
            title = article.get('title', f'Article {i+1}')
            content = article.get('content', '')[:300]  # More content for better grouping
            date = article.get('analysis', {}).get('published_date', 'Unknown date')
            source = article.get('source', 'Unknown source')
            article_summaries.append(f"Article {i+1}: [{date}] {title} - {content}... (Source: {source})")
        
        # Enhanced grouping prompt with clearer instructions
        grouping_prompt = f"""Group these unique articles about {company_name} by thematic relationships.

GROUPING STRATEGY (prioritized):
1. REGULATORY BODY: Group by same regulator (SEBI, RBI, CBI, IRDAI, MCA, ED, IT, etc.)
2. VIOLATION TYPE: Group similar violations (disclosure, fraud, compliance, governance)
3. BUSINESS AREA: Group by company division (ports, power, green energy, etc.)
4. CHRONOLOGICAL: Group articles about same ongoing investigation/case
5. SEVERITY: Group major penalties/actions vs minor violations

GROUPING RULES:
- Minimum 2 articles per group (unless only 1 remains)
- Maximum 4 articles per group for readability
- Create meaningful theme names (e.g., "SEBI Disclosure Violations", "CBI Financial Investigations")
- If articles are too diverse, keep some as individual rather than force grouping

JSON FORMAT:
{{
  "groups": [
    {{
      "theme": "SEBI Regulatory Actions",
      "article_indices": [1, 3],
      "reasoning": "Both involve SEBI penalties for disclosure violations"
    }},
    {{
      "theme": "Individual",
      "article_indices": [2],
      "reasoning": "Unique incident not related to other articles"
    }}
  ]
}}

Return ONLY valid JSON, no other text."""
        
        articles_text = "\n".join(article_summaries)
        human_prompt = f"Group these {len(articles)} unique {company_name} articles by theme:\n{articles_text}"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content=grouping_prompt),
                    HumanMessage(content=human_prompt)
                ]
                print(f"   💬 Sending grouping request to LLM (attempt {attempt + 1}/{max_retries})...")
                response = await self.llm.ainvoke(messages)
                
                raw_content = response.content.strip()
                print(f"   ✅ Grouping response: {raw_content[:100]}...")
                
                # Clean up response
                if raw_content.startswith('```json'):
                    raw_content = raw_content.replace('```json', '').replace('```', '').strip()
                elif raw_content.startswith('```'):
                    raw_content = raw_content.replace('```', '').strip()
                
                import json
                grouping_result = json.loads(raw_content)
                
                if 'groups' not in grouping_result:
                    raise ValueError("Invalid JSON format - missing 'groups' key")
                
                # Validate and convert to our format
                groups = []
                used_indices = set()
                
                for group_data in grouping_result.get('groups', []):
                    indices = group_data.get('article_indices', [])
                    theme = group_data.get('theme', 'Related Issues')
                    reasoning = group_data.get('reasoning', '')
                    
                    # Validate indices
                    valid_indices = [i for i in indices if 0 < i <= len(articles) and (i-1) not in used_indices]
                    
                    if valid_indices:
                        group_articles = [articles[i-1] for i in valid_indices]
                        groups.append({
                            'theme': theme,
                            'articles': group_articles
                        })
                        used_indices.update(i-1 for i in valid_indices)
                        print(f"   📁 Created group '{theme}' with {len(group_articles)} articles")
                        if reasoning:
                            print(f"      Reasoning: {reasoning}")
                
                # Handle any remaining ungrouped articles
                remaining_articles = [article for i, article in enumerate(articles) if i not in used_indices]
                if remaining_articles:
                    print(f"   📄 Adding {len(remaining_articles)} ungrouped articles individually")
                    for article in remaining_articles:
                        groups.append({
                            'theme': 'Individual',
                            'articles': [article]
                        })
                
                # Validate final result
                total_articles_grouped = sum(len(group['articles']) for group in groups)
                if total_articles_grouped != len(articles):
                    print(f"   ⚠️ Article count mismatch: {total_articles_grouped} grouped vs {len(articles)} original")
                    # Continue anyway but log the discrepancy
                
                print(f"   📊 Successfully created {len(groups)} groups from {len(articles)} unique articles")
                return groups
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parsing failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"   ⚠️ Raw response: {raw_content[:200]}...")
            except Exception as e:
                print(f"   ⚠️ Grouping failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    break
        
        # Robust fallback: Intelligent grouping based on simple rules
        print(f"   🔄 LLM grouping failed, falling back to rule-based grouping")
        return self._fallback_grouping(articles, company_name)
    
    def _fallback_grouping(self, articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
        """Fallback grouping using simple rules when LLM fails"""
        
        groups = []
        grouped_articles = set()
        
        # Group by regulatory body keywords
        regulators = {
            'SEBI': ['sebi', 'securities', 'exchange', 'board'],
            'RBI': ['rbi', 'reserve', 'bank', 'monetary'],
            'CBI': ['cbi', 'central', 'bureau', 'investigation'],
            'IRDAI': ['irdai', 'insurance', 'regulatory'],
            'MCA': ['mca', 'corporate', 'affairs', 'ministry'],
            'ED': ['enforcement', 'directorate', 'ed'],
            'IT': ['income', 'tax', 'department']
        }
        
        for regulator, keywords in regulators.items():
            regulator_articles = []
            for i, article in enumerate(articles):
                if i in grouped_articles:
                    continue
                    
                title_content = (article.get('title', '') + ' ' + article.get('content', '')[:200]).lower()
                if any(keyword in title_content for keyword in keywords):
                    regulator_articles.append(article)
                    grouped_articles.add(i)
            
            if len(regulator_articles) >= 2:
                groups.append({
                    'theme': f'{regulator} Regulatory Actions',
                    'articles': regulator_articles
                })
                print(f"   📁 Fallback: Grouped {len(regulator_articles)} articles under {regulator}")
            elif len(regulator_articles) == 1:
                groups.append({
                    'theme': 'Individual',
                    'articles': regulator_articles
                })
        
        # Add remaining articles as individual
        for i, article in enumerate(articles):
            if i not in grouped_articles:
                groups.append({
                    'theme': 'Individual',
                    'articles': [article]
                })
        
        print(f"   📊 Fallback grouping created {len(groups)} groups")
        return groups
    
    async def _create_grouped_finding(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """Create grouped finding with summary and bullet points"""
        
        articles = group['articles']
        theme = group['theme']
        
        # Sort articles within the group chronologically (earliest to latest) to show progression
        def get_article_date(article):
            """Get article date for sorting within group"""
            date_str = article.get('analysis', {}).get('published_date', '')
            if not date_str:
                return datetime.min
            try:
                if 'T' in date_str:
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%y', '%d-%b-%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
            return datetime.min
        
        # Sort articles from earliest to latest within the group
        articles = sorted(articles, key=get_article_date)
        print(f"   📅 Sorted {len(articles)} articles within group '{theme}' chronologically (earliest to latest)")
        
        # Debug: show the sorted dates within the group
        for i, article in enumerate(articles):
            date_str = article.get('analysis', {}).get('published_date', 'No date')
            title_preview = article.get('title', 'No title')[:50] + '...' if len(article.get('title', '')) > 50 else article.get('title', 'No title')
            print(f"      📅 Article {i+1}: {date_str} - {title_preview}")
        
        # Generate group summary
        summary = await self._generate_group_summary(articles, theme)
        
        # Generate bullet points for each article
        bullet_points = []
        source_entries = []
        dates = []
        reliabilities = []
        
        for article in articles:
            bullet_content = await self._generate_bullet_point(article)
            # Use blue rectangle instead of black circle
            bullet_points.append(f"<font color='blue'>▪</font> {bullet_content}")
            
            # Create proper source entry with URL if available
            source_name = self._extract_source_name(article)
            source_url = article.get('url', '')
            if source_url and source_url.strip():
                source_entry = f'<link href="{source_url}" color="blue"><u>{source_name}</u></link>'
            else:
                source_entry = f'<font color="blue"><u>{source_name}</u></font>'
            source_entries.append(source_entry)
            
            analysis = article.get('analysis', {})
            dates.append(self._parse_date(analysis.get('published_date', '')))
            reliabilities.append(self._determine_reliability(article))
        
        # Combine summary and bullet points with proper formatting
        formatted_bullets = []
        for bullet in bullet_points:
            # Format each bullet with proper indentation and spacing
            formatted_bullets.append(f"<br/>&nbsp;&nbsp;&nbsp;{bullet}")
        
        # Combine with clear separation
        key_finding = f"{summary}<br/><br/>" + "<br/>".join(formatted_bullets)
        
        # Reverse display order for metadata columns (latest first) while keeping Key Findings chronological
        dates.reverse()
        source_entries.reverse()
        reliabilities.reverse()
        
        # Get latest date for sorting (to ensure groups are ordered by most recent article)
        raw_dates = [article.get('analysis', {}).get('published_date', '') for article in articles]
        # Filter out empty dates and find latest (most recent)
        valid_dates = [date for date in raw_dates if date and date.strip()]
        latest_date = max(valid_dates) if valid_dates else ''
        
        return {
            'key_finding': key_finding,
            'source': '<br/><br/>'.join(source_entries),  # Use HTML formatted sources
            'source_url': '',
            'report_date': "\n".join(dates),
            'reliability': "\n".join(reliabilities),
            'raw_date': latest_date,
        }
    
    async def _generate_group_summary(self, articles: List[Dict[str, Any]], theme: str) -> str:
        """Generate 1-2 sentence summary for article group"""
        
        titles = [article.get('title', '') for article in articles]
        contents = [article.get('content', '')[:300] for article in articles]
        
        system_prompt = """Create a chronological summary showing how ADVERSE/NEGATIVE issues developed over time for adverse media reporting.

FOCUS ON ADVERSE CONTENT ONLY:
- Regulatory violations, penalties, fines, investigations
- Legal issues, compliance failures, governance problems
- Fraud allegations, financial misconduct, operational failures
- Penalties imposed, enforcement actions, show cause notices

IGNORE POSITIVE/NEUTRAL CONTENT:
- Support announcements, backing declarations, confidence statements
- Investment commitments, partnership announcements
- Business expansions, growth plans, market opportunities

Requirements:
- Show ADVERSE progression: how issue started → developed → resolved/ongoing
- Include month/year when ADVERSE events occurred
- Use abbreviations: SEBI, RBI, CBI, IRDAI, etc.
- Keep factual, concise (2-3 sentences max)
- Focus ONLY on negative/adverse timeline

Example: "In Jan 2024, SEBI initiated investigation into disclosure violations. By Mar 2024, penalties of INR 50 crore were imposed. The matter resulted in compliance improvements by Jun 2024."""
        
        # Sort articles by date for chronological summary
        article_data = []
        for i, article in enumerate(articles):
            date = article.get('analysis', {}).get('published_date', '')
            article_data.append({
                'index': i,
                'title': titles[i],
                'content': contents[i],
                'date': date
            })
        
        # Sort by date for chronological order in summary
        article_data.sort(key=lambda x: self._parse_date_for_sorting(x['date']))
        
        articles_text = "\n".join([f"Date: {item['date']}\nTitle: {item['title']}\nContent: {item['content']}" for item in article_data])
        human_prompt = f"Theme: {theme}\n\nArticles (in chronological order):\n{articles_text}\n\nCreate chronological progression summary:"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
        except Exception:
            return f"Multiple adverse events related to {theme.lower()}."
    
    async def _generate_bullet_point(self, article: Dict[str, Any]) -> str:
        """Generate 60-90 word bullet point for article"""
        
        title = article.get('title', '')
        content = article.get('content', '')
        analysis = article.get('analysis', {})
        published_date = analysis.get('published_date', '')
        
        system_prompt = """Create a 60-90 word bullet point starting with "On [date]" format focusing EXCLUSIVELY on ADVERSE/NEGATIVE content.

FOCUS ON ADVERSE CONTENT ONLY:
- Regulatory violations, penalties, fines, investigations
- Legal issues, compliance failures, governance problems
- Fraud allegations, financial misconduct, operational failures
- Penalties imposed, enforcement actions, show cause notices
- Criminal charges, lawsuits, regulatory sanctions

IGNORE POSITIVE/NEUTRAL CONTENT:
- Support announcements, backing declarations, confidence statements
- Investment commitments, partnership announcements
- Business growth, expansions, market opportunities

DATE FORMAT: Use "DD Month, YYYY" format (e.g., "15 January, 2024")

Example: "On 15 January, 2024, SEBI imposed penalties of INR 25 crore for disclosure violations, citing inadequate compliance with reporting requirements and potential impact on investor transparency."""
        
        human_prompt = f"Title: {title}\nDate: {published_date}\nContent: {content[:500]}\n\nCreate bullet point:"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.ainvoke(messages)
            bullet = response.content.strip()
            
            # Ensure reasonable length
            if len(bullet) > 400:
                truncated = bullet[:397]
                last_period = truncated.rfind('.')
                if last_period > 200:
                    bullet = bullet[:last_period + 1]
                else:
                    bullet = truncated + '..'
            
            return bullet
        except Exception:
            return f"On {published_date}, adverse event occurred related to {title[:50]}..."
    
    def _extract_source_name(self, result: Dict[str, Any]) -> str:
        """Extract proper source name from analysis results or article metadata"""
        
        # Try to extract source name from various locations in the analysis pipeline data structure
        source_name = ""
        
        # 1. First try from the direct source field (coming from article analysis)
        source_name = result.get('source', '')
        
        # 2. If source is 'extracted_content', 'direct', or empty, try other locations
        if not source_name or source_name in ['extracted_content', 'direct', '']:
            # Try from content_metadata (analysis pipeline structure)
            content_metadata = result.get('content_metadata', {})
            source_name = content_metadata.get('source', '')
        
        # 3. If still no source, try to extract from URL
        if not source_name or source_name in ['extracted_content', 'direct', '']:
            url = result.get('url', '')
            if url:
                # Extract domain name and convert to readable format
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    
                    # Remove www prefix
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    
                    # Map common domains to readable names (expanded mapping)
                    domain_mapping = {
                        'timesofindia.indiatimes.com': 'Times of India',
                        'economictimes.indiatimes.com': 'The Economic Times',
                        'm.economictimes.com': 'The Economic Times',
                        'business-standard.com': 'Business Standard',
                        'financialexpress.com': 'The Financial Express',
                        'livemint.com': 'Mint',
                        'hindustantimes.com': 'Hindustan Times',
                        'indianexpress.com': 'The Indian Express',
                        'moneycontrol.com': 'Moneycontrol',
                        'businesstoday.in': 'Business Today',
                        'cnbctv18.com': 'CNBC TV18',
                        'ndtv.com': 'NDTV',
                        'thehindu.com': 'The Hindu',
                        'indiatoday.in': 'India Today',
                        'newindianexpress.com': 'The New Indian Express',
                        'deccanherald.com': 'Deccan Herald',
                        'tribuneindia.com': 'The Tribune',
                        'freepressjournal.in': 'Free Press Journal',
                        'outlookindia.com': 'Outlook India',
                        'news18.com': 'News18',
                        'abp.live': 'ABP Live',
                        'republicworld.com': 'Republic World',
                        'zeenews.india.com': 'Zee News',
                        'reuters.com': 'Reuters',
                        'bloomberg.com': 'Bloomberg',
                        'thehindubusinessline.com': 'The Hindu Business Line',
                        'firstpost.com': 'Firstpost',
                        'scroll.in': 'Scroll.in',
                        'theprint.in': 'The Print',
                        'wire.in': 'The Wire',
                        'mathrubhumi.com': 'Mathrubhumi',
                        'pgurus.com': 'PGurus',
                        'scconline.com': 'SCC Online',
                        'opindia.com': 'OpIndia',
                        'kalingatv.com': 'Kalinga TV',
                        'tradebrains.in': 'Trade Brains'
                    }
                    
                    source_name = domain_mapping.get(domain, domain.replace('.com', '').replace('.in', '').title())
                    
                except Exception as e:
                    print(f"⚠️ Error extracting source from URL: {e}")
                    source_name = 'Unknown Source'
        
        # Final fallback - if source is still problematic, set to Unknown
        if source_name in ['extracted_content', 'direct', '']:
            source_name = 'Unknown Source'
        
        print(f"🏢 Extracted source name: '{source_name}' from {result.get('url', 'No URL')[:60]}...")
        return source_name
    
    def _sort_findings_by_credibility_and_date(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort findings by credibility/relevance, then by date (latest first)"""
        
        def get_credibility_score(finding):
            """Calculate credibility score based on source reliability and content relevance"""
            score = 0
            
            # Source reliability scoring
            reliability = finding.get('reliability', 'Medium')
            if '\n' in reliability:  # Multiple sources
                reliabilities = reliability.split('\n')
                high_count = sum(1 for r in reliabilities if r.strip().lower() == 'high')
                medium_count = sum(1 for r in reliabilities if r.strip().lower() == 'medium')
                score += (high_count * 10) + (medium_count * 5)
            else:
                if reliability.lower() == 'high':
                    score += 10
                elif reliability.lower() == 'medium':
                    score += 5
            
            # Content relevance scoring (regulatory actions get higher scores)
            key_finding = finding.get('key_finding', '').lower()
            
            # High relevance terms
            high_relevance = ['sebi', 'rbi', 'cbi', 'irdai', 'penalty', 'fine', 'investigation', 
                            'violation', 'fraud', 'bribery', 'corruption', 'arrest', 'conviction']
            score += sum(3 for term in high_relevance if term in key_finding)
            
            # Medium relevance terms
            medium_relevance = ['regulatory', 'compliance', 'audit', 'inspection', 'notice',
                              'warning', 'suspension', 'cancellation']
            score += sum(2 for term in medium_relevance if term in key_finding)
            
            return score
        
        def get_sort_date(finding):
            """Extract date for sorting (latest first)"""
            raw_date = finding.get('raw_date', '')
            if not raw_date:
                return datetime.min  # Put articles without dates at the end
            
            try:
                if 'T' in raw_date:
                    return datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                
                # Try various date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%y', '%d-%b-%Y', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        return datetime.strptime(raw_date, fmt)
                    except:
                        continue
            except Exception as e:
                print(f"⚠️ Failed to parse date '{raw_date}': {e}")
            
            return datetime.min  # Put unparseable dates at the end
        
        # Sort primarily by date (latest first), then by credibility score as secondary factor
        sorted_findings = sorted(findings, 
                               key=lambda x: (get_sort_date(x), get_credibility_score(x)), 
                               reverse=True)
        
        print(f"📅 Sorted {len(sorted_findings)} findings by credibility and date (latest groups/articles first)")
        
        # Debug print the first few dates to verify group ordering
        for i, finding in enumerate(sorted_findings[:3]):
            raw_date = finding.get('raw_date', '')
            key_finding_preview = finding.get('key_finding', '')[:100] + '...' if len(finding.get('key_finding', '')) > 100 else finding.get('key_finding', '')
            print(f"   📅 Finding {i+1} date: {raw_date} - {key_finding_preview}")
        return sorted_findings
    
    def _parse_date_for_sorting(self, date_str: str) -> datetime:
        """Parse date string for chronological sorting"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try various date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%Y', '%B %d, %Y']:
                try:
                    return datetime.strptime(date_str[:10] if 'T' in date_str else date_str, fmt)
                except:
                    continue
            
            # Try ISO format
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        except Exception as e:
            print(f"Date parsing error for '{date_str}': {e}")
        
        return datetime.now()
    
    def _determine_time_period(self, findings: List[Dict[str, Any]]) -> str:
        """Determine time period covered by the findings"""
        
        if not findings:
            return "No time period specified"
        
        dates = []
        for finding in findings:
            raw_date = finding.get('raw_date', '')
            if raw_date:
                try:
                    if 'T' in raw_date:
                        dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                        dates.append(dt)
                    else:
                        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                            try:
                                dt = datetime.strptime(raw_date, fmt)
                                dates.append(dt)
                                break
                            except:
                                continue
                except Exception:
                    continue
        
        if not dates:
            return "Recent period"
        
        # Get date range
        min_date = min(dates)
        max_date = max(dates)
        
        # Format time period
        if min_date.date() == max_date.date():
            return f"Single day analysis: {max_date.strftime('%d %B %Y')}"
        elif (max_date - min_date).days <= 31:
            return f"Period: {min_date.strftime('%d %B %Y')} to {max_date.strftime('%d %B %Y')}"
        else:
            return f"Period: {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}"
    
    def _parse_date(self, date_str: str) -> str:
        """Parse and format date string to consistent 'DD Month, YYYY' format"""
        if not date_str:
            return datetime.now().strftime('%d %B, %Y')
        
        try:
            # Try to parse ISO format
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%d %B, %Y')
            
            # Try other common formats
            date_formats = [
                '%Y-%m-%d',      # 2024-01-15
                '%d/%m/%Y',      # 15/01/2024
                '%m/%d/%Y',      # 01/15/2024
                '%d-%b-%Y',      # 15-Jan-2024
                '%d %b %Y',      # 15 Jan 2024
                '%d-%m-%Y',      # 15-01-2024
                '%Y-%m-%dT%H:%M:%S',  # ISO with time
                '%B %d, %Y',     # January 15, 2024
                '%d %B %Y'       # 15 January 2024
            ]
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d %B, %Y')
                except ValueError:
                    continue
                    
            # If no format worked, try to extract just the date part and retry
            if len(date_str) >= 10:
                date_part = date_str[:10]
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        dt = datetime.strptime(date_part, fmt)
                        return dt.strftime('%d %B, %Y')
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Date parsing error for '{date_str}': {e}")
        
        # Final fallback - return original if all parsing fails
        return date_str if date_str else 'Unknown Date'
    
    def _determine_reliability(self, result: Dict[str, Any]) -> str:
        """Determine source reliability based on confidence score and source"""
        
        # Extract confidence from the correct nested structure
        analysis = result.get('analysis', {})
        metadata = analysis.get('metadata', {})
        confidence = metadata.get('confidence_score', 0.5)
        
        source = result.get('source', '').lower()
        
        # High reliability sources
        high_reliability_sources = ['reuters', 'bloomberg', 'financial times', 'wall street journal', 
                                   'economic times', 'business standard', 'mint', 'moneycontrol']
        
        # Check if source is high reliability
        for high_source in high_reliability_sources:
            if high_source in source:
                return 'High'
        
        # Based on confidence score
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.5:
            return 'Medium'
        else:
            return 'Low'


# Example usage
def create_sample_report():
    """Create a sample adverse media report"""
    generator = AdverseMediaPDFReport()
    
    # Sample adverse findings matching your image
    sample_findings = [
        {
            'key_finding': ('NIACL has a high public profile. Online references to the company are primarily in relation to its business activities. '
                          'NIACL has been subject of adverse press in relation to various penalties amounting to approximately INR 25 lakhs imposed by Insurance Regulatory and Development Authority of India (IRDAI) between the period 2016-2024 for violation of provisions of the Insurance Act and mediclaim policy guidelines '
                          '(considering the size of the NIACL, the amounts involved are not material impact to the NIACL); employees being accused and imprisoned by Central Bureau of Investigation (CBI) for fraudulent claims caused huge losses to NIACL, accepting bribe, among others; settlement with Securities and Exchange Board of India (SEBI) in relation to insider trading case; business improvement order issued by Japanese Financial Services Agency; being cited on defaulter list of Employee State Insurance Corporation (ESIC), among other issues. Details of the issues have been summarized below:'),
            'source': 'IRDAI Order',
            'source_url': 'https://www.irdai.gov.in',
            'report_date': '18-Sep-19',
            'reliability': 'High'
        },
        {
            'key_finding': 'According to online sources, NIACL was subject of various penalties imposed by Insurance Regulatory and Development Authority of India (IRDAI) for violation of provision of the Insurance Acts and mediclaim policy guidelines. Some of which are listed below:',
            'source': 'Cafe Mutual',
            'source_url': 'https://cafemutual.com',
            'report_date': '18-Sep-19',
            'reliability': 'Medium'
        },
        {
            'key_finding': 'INR 300,000 in September 2019 for alleged certain violations of Regulations; INR 1,00,000 in 2018 in relation to delay in making an offer by NIACL after receipt of survey report which is in violation of the rule of the NIACL had settled claim after six months of receiving the survey report in a few instance after undertaking to rule of 30 days.',
            'source': 'IRDAI Order',
            'source_url': '',
            'report_date': '17-Mar-16',
            'reliability': 'High'
        },
        {
            'key_finding': 'INR 15,00,000 in March 2016 for alleged certain violation of provisions of the Insurance Act, 1938. It had been observed that NIACL had not exercised internal payouts over and above commission to Maruti Insurance Business Centre Limited on the motor premium procured through the two corporate agents.',
            'source': 'Times of India',
            'source_url': '',
            'report_date': '10-May-16',
            'reliability': 'Medium'
        },
        {
            'key_finding': 'INR 500,000 in January 2016 for alleged violation of Mediclaim policy guidelines i.e., accusing NIACL of rejecting cashless claim on grounds that can t be applied to group medical policies.',
            'source': 'Business Standard',
            'source_url': '',
            'report_date': '21-Jan-16',
            'reliability': 'Medium'
        }
    ]
    
    # Generate report
    output_path = generator.generate_report(
        company_name="New India Assurance Company Limited (NIACL)",
        adverse_findings=sample_findings,
        output_path="sample_adverse_media_report.pdf",
        report_metadata={
            'total_articles_analyzed': 150,
            'adverse_articles_found': 5
        }
    )
    
    print(f"✅ Sample report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_report()



"""
Advanced Chart and Image Analysis Module
Analyzes charts, graphs, diagrams, and visual content
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """Advanced chart and visual content analyzer"""
    
    def __init__(self):
        self.chart_types = {
            'bar_chart': self._analyze_bar_chart,
            'line_chart': self._analyze_line_chart,
            'pie_chart': self._analyze_pie_chart,
            'scatter_plot': self._analyze_scatter_plot,
            'table': self._analyze_table_image,
            'diagram': self._analyze_diagram,
            'flowchart': self._analyze_flowchart
        }
    
    def analyze_image_content(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive image analysis including chart detection and data extraction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Detailed analysis results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            analysis_result = {
                "image_path": image_path,
                "image_type": "unknown",
                "chart_type": None,
                "extracted_data": {},
                "text_content": "",
                "visual_elements": {},
                "data_points": [],
                "insights": [],
                "confidence_scores": {}
            }
            
            # Detect image type and chart type
            image_type, chart_type = self._detect_chart_type(image)
            analysis_result["image_type"] = image_type
            analysis_result["chart_type"] = chart_type
            
            # Extract text from image (OCR)
            text_content = self._extract_text_from_image(image_path)
            analysis_result["text_content"] = text_content
            
            # Analyze visual elements
            visual_elements = self._analyze_visual_elements(image)
            analysis_result["visual_elements"] = visual_elements
            
            # Perform chart-specific analysis
            if chart_type and chart_type in self.chart_types:
                chart_analysis = self.chart_types[chart_type](image, text_content)
                analysis_result["extracted_data"] = chart_analysis
            
            # Extract data points and patterns
            data_points = self._extract_data_points(image, chart_type)
            analysis_result["data_points"] = data_points
            
            # Generate insights
            insights = self._generate_insights(analysis_result)
            analysis_result["insights"] = insights
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {"error": str(e), "image_path": image_path}
    
    def _detect_chart_type(self, image: np.ndarray) -> Tuple[str, Optional[str]]:
        """Detect the type of chart or visual content"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze shapes and patterns
            if self._detect_circular_patterns(contours):
                return "chart", "pie_chart"
            elif self._detect_rectangular_bars(contours):
                return "chart", "bar_chart"
            elif self._detect_line_patterns(edges):
                return "chart", "line_chart"
            elif self._detect_grid_pattern(edges):
                return "chart", "table"
            elif self._detect_scatter_points(contours):
                return "chart", "scatter_plot"
            elif self._detect_flowchart_elements(contours):
                return "diagram", "flowchart"
            else:
                return "image", "diagram"
                
        except Exception as e:
            logger.error(f"Error detecting chart type: {e}")
            return "unknown", None
    
    def _detect_circular_patterns(self, contours) -> bool:
        """Detect circular patterns (pie charts)"""
        circles = 0
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly circular (many vertices)
            if len(approx) > 8:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # Reasonably circular
                        circles += 1
        
        return circles > 0
    
    def _detect_rectangular_bars(self, contours) -> bool:
        """Detect rectangular bars (bar charts)"""
        rectangles = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's rectangular (4 vertices)
            if len(approx) == 4:
                rectangles += 1
        
        # Bar charts typically have multiple rectangles
        return rectangles >= 3
    
    def _detect_line_patterns(self, edges: np.ndarray) -> bool:
        """Detect line patterns (line charts)"""
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Check for connected line segments (typical in line charts)
            return len(lines) > 5
        return False
    
    def _detect_grid_pattern(self, edges: np.ndarray) -> bool:
        """Detect grid patterns (tables)"""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Check if both horizontal and vertical lines exist
        h_sum = np.sum(horizontal_lines > 0)
        v_sum = np.sum(vertical_lines > 0)
        
        return h_sum > 100 and v_sum > 100  # Threshold for grid detection
    
    def _detect_scatter_points(self, contours) -> bool:
        """Detect scatter plot points"""
        small_circles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # Look for small circular shapes (scatter points)
            if 10 < area < 100:  # Small areas typical of scatter points
                small_circles += 1
        
        return small_circles > 10  # Multiple small points
    
    def _detect_flowchart_elements(self, contours) -> bool:
        """Detect flowchart elements (boxes, diamonds, etc.)"""
        shapes = {"rectangles": 0, "diamonds": 0, "circles": 0}
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                shapes["rectangles"] += 1
            elif len(approx) == 4:  # Diamond shapes
                shapes["diamonds"] += 1
            elif len(approx) > 8:  # Circular shapes
                shapes["circles"] += 1
        
        # Flowcharts typically have multiple different shapes
        return sum(shapes.values()) > 3
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Use PIL for better OCR results
            pil_image = Image.open(image_path)
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(2.0)
            
            # Extract text
            text = pytesseract.image_to_string(enhanced_image, config='--psm 6')
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _analyze_visual_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze visual elements like colors, shapes, etc."""
        try:
            # Color analysis
            colors = self._analyze_colors(image)
            
            # Shape analysis
            shapes = self._count_shapes(image)
            
            # Size analysis
            height, width = image.shape[:2]
            
            return {
                "dimensions": {"width": width, "height": height},
                "dominant_colors": colors,
                "shape_count": shapes,
                "aspect_ratio": width / height
            }
        except Exception as e:
            logger.error(f"Error analyzing visual elements: {e}")
            return {}
    
    def _analyze_colors(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze dominant colors in the image"""
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Use k-means to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 5  # Number of dominant colors to find
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and get unique colors
            centers = np.uint8(centers)
            
            # Count pixels for each color
            unique, counts = np.unique(labels, return_counts=True)
            
            dominant_colors = []
            for i, color in enumerate(centers):
                percentage = (counts[i] / len(labels)) * 100
                dominant_colors.append({
                    "color_rgb": color.tolist(),
                    "percentage": round(percentage, 2)
                })
            
            # Sort by percentage
            dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
            return dominant_colors[:3]  # Top 3 colors
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return []
    
    def _count_shapes(self, image: np.ndarray) -> Dict[str, int]:
        """Count different types of shapes in the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = {"rectangles": 0, "circles": 0, "triangles": 0, "other": 0}
            
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 3:
                    shapes["triangles"] += 1
                elif len(approx) == 4:
                    shapes["rectangles"] += 1
                elif len(approx) > 8:
                    shapes["circles"] += 1
                else:
                    shapes["other"] += 1
            
            return shapes
        except Exception as e:
            logger.error(f"Error counting shapes: {e}")
            return {}
    
    def _analyze_bar_chart(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze bar chart specific features"""
        return {
            "chart_type": "bar_chart",
            "axes_detected": self._detect_axes(image),
            "bar_count": self._count_bars(image),
            "orientation": self._detect_bar_orientation(image),
            "title": self._extract_title_from_text(text),
            "labels": self._extract_labels_from_text(text)
        }
    
    def _analyze_line_chart(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze line chart specific features"""
        return {
            "chart_type": "line_chart",
            "line_count": self._count_lines(image),
            "trend": self._analyze_trend(image),
            "data_points": self._estimate_data_points(image),
            "title": self._extract_title_from_text(text),
            "axes_labels": self._extract_axes_labels_from_text(text)
        }
    
    def _analyze_pie_chart(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze pie chart specific features"""
        return {
            "chart_type": "pie_chart",
            "segments": self._count_pie_segments(image),
            "percentages": self._extract_percentages_from_text(text),
            "legend": self._extract_legend_from_text(text),
            "title": self._extract_title_from_text(text)
        }
    
    def _analyze_scatter_plot(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze scatter plot specific features"""
        return {
            "chart_type": "scatter_plot",
            "point_count": self._count_scatter_points(image),
            "clusters": self._detect_clusters(image),
            "correlation": self._estimate_correlation(image),
            "title": self._extract_title_from_text(text)
        }
    
    def _analyze_table_image(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze table in image format"""
        return {
            "content_type": "table",
            "rows_detected": self._count_table_rows(image),
            "columns_detected": self._count_table_columns(image),
            "extracted_text": text,
            "structure": self._analyze_table_structure(image)
        }
    
    def _analyze_diagram(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze general diagrams"""
        return {
            "content_type": "diagram",
            "elements": self._count_diagram_elements(image),
            "connections": self._detect_connections(image),
            "text_regions": self._locate_text_regions(image),
            "extracted_text": text
        }
    
    def _analyze_flowchart(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze flowchart specific features"""
        return {
            "content_type": "flowchart",
            "decision_nodes": self._count_decision_nodes(image),
            "process_nodes": self._count_process_nodes(image),
            "connections": self._detect_flowchart_connections(image),
            "flow_direction": self._detect_flow_direction(image),
            "extracted_text": text
        }
    
    def _extract_data_points(self, image: np.ndarray, chart_type: Optional[str]) -> List[Dict[str, Any]]:
        """Extract approximate data points from charts"""
        try:
            if chart_type == "line_chart":
                return self._extract_line_chart_points(image)
            elif chart_type == "bar_chart":
                return self._extract_bar_chart_values(image)
            elif chart_type == "scatter_plot":
                return self._extract_scatter_points(image)
            else:
                return []
        except Exception as e:
            logger.error(f"Error extracting data points: {e}")
            return []
    
    def _generate_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate insights based on the analysis"""
        insights = []
        
        try:
            chart_type = analysis_result.get("chart_type")
            extracted_data = analysis_result.get("extracted_data", {})
            
            if chart_type == "bar_chart":
                bar_count = extracted_data.get("bar_count", 0)
                if bar_count > 0:
                    insights.append(f"Bar chart contains {bar_count} data categories")
                    
            elif chart_type == "line_chart":
                trend = extracted_data.get("trend", "")
                if trend:
                    insights.append(f"Line chart shows {trend} trend")
                    
            elif chart_type == "pie_chart":
                segments = extracted_data.get("segments", 0)
                if segments > 0:
                    insights.append(f"Pie chart divided into {segments} segments")
            
            # General insights
            text_content = analysis_result.get("text_content", "")
            if text_content:
                word_count = len(text_content.split())
                insights.append(f"Contains {word_count} words of text")
            
            visual_elements = analysis_result.get("visual_elements", {})
            if visual_elements.get("dominant_colors"):
                color_count = len(visual_elements["dominant_colors"])
                insights.append(f"Uses {color_count} dominant colors")
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    # Placeholder methods for specific analysis functions
    # These would contain more sophisticated image processing logic
    
    def _detect_axes(self, image): return True
    def _count_bars(self, image): return 5
    def _detect_bar_orientation(self, image): return "vertical"
    def _count_lines(self, image): return 2
    def _analyze_trend(self, image): return "increasing"
    def _estimate_data_points(self, image): return 10
    def _count_pie_segments(self, image): return 4
    def _count_scatter_points(self, image): return 25
    def _detect_clusters(self, image): return 3
    def _estimate_correlation(self, image): return "positive"
    def _count_table_rows(self, image): return 5
    def _count_table_columns(self, image): return 3
    def _analyze_table_structure(self, image): return {"headers": True, "borders": True}
    def _count_diagram_elements(self, image): return 8
    def _detect_connections(self, image): return 5
    def _locate_text_regions(self, image): return []
    def _count_decision_nodes(self, image): return 2
    def _count_process_nodes(self, image): return 5
    def _detect_flowchart_connections(self, image): return 7
    def _detect_flow_direction(self, image): return "top_to_bottom"
    def _extract_line_chart_points(self, image): return []
    def _extract_bar_chart_values(self, image): return []
    def _extract_scatter_points(self, image): return []
    
    def _extract_title_from_text(self, text): 
        lines = text.split('\n')
        return lines[0] if lines else ""
    
    def _extract_labels_from_text(self, text): 
        return [word for word in text.split() if len(word) > 2]
    
    def _extract_axes_labels_from_text(self, text): 
        return {"x_axis": "", "y_axis": ""}
    
    def _extract_percentages_from_text(self, text): 
        import re
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        return [float(p.replace('%', '')) for p in percentages]
    
    def _extract_legend_from_text(self, text): 
        return text.split('\n')[1:] if '\n' in text else []

# Global chart analyzer instance
chart_analyzer = ChartAnalyzer()
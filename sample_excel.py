"""
FontDiffuser Batch Processing with Robust Font Rendering
Fixed the cropping bug in ttf2im and added safety checks
"""

import os
import sys
import json
import time
import warnings
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import ast
import torch
import numpy as np
import cv2
from PIL import Image
import pygame
import pygame.freetype
from fontTools.ttLib import TTFont
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Suppress warnings
warnings.filterwarnings('ignore')

# Import FontDiffuser components
try:
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder
    )
    from utils import (
        save_args_to_yaml,
        save_single_image,
        save_image_with_content_style
    )
except ImportError as e:
    print(f"Error importing FontDiffuser modules: {e}")
    print("Please ensure the required modules are in your Python path")
    sys.exit(1)


class FontRenderer:
    """
    Robust font renderer with fixed ttf2im and additional safety checks
    """
    
    def __init__(self, font_size: int = 256, canvas_size: int = 256):
        """
        Args:
            font_size: Font size for rendering
            canvas_size: Size of output image (should be >= font_size)
        """
        self.font_size = font_size
        self.canvas_size = canvas_size
        
        # Initialize pygame for font rendering
        if not pygame.get_init():
            pygame.init()
    
    def ttf2im_fixed(self, font, char: str, debug: bool = False) -> Optional[Image.Image]:
        """
        Fixed version of ttf2im that properly centers characters without cropping
        
        Args:
            font: pygame.freetype.Font object
            char: Character to render
            debug: Whether to save debug images
            
        Returns:
            PIL Image of rendered character or None on failure
        """
        try:
            # Create a transparent surface larger than needed
            padding = 50  # Extra padding to prevent edge cropping
            surface_size = self.canvas_size + padding * 2
            surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            surface.fill((0, 0, 0, 0))
            
            # Render character in center
            text_surface, text_rect = font.render(
                char,
                fgcolor=(0, 0, 0),  # Black color
                size=self.font_size
            )
            
            # Center on surface
            text_rect.center = (surface_size // 2, surface_size // 2)
            surface.blit(text_surface, text_rect)
            
            # Get alpha channel
            imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
            imo = 255 - np.array(Image.fromarray(imo))
            
            # Create white background
            bg = np.full((self.canvas_size, self.canvas_size), 255, dtype=np.uint8)
            
            # Resize if needed, preserving aspect ratio
            h, w = imo.shape[:2]
            
            # Debug: Save the raw rendered image
            if debug:
                debug_raw = Image.fromarray(imo).convert('RGB')
                debug_raw.save(f"debug_raw_{char}.png")
            
            # Find bounding box of non-white pixels
            # Threshold to handle anti-aliasing
            threshold = 10
            binary = imo < (255 - threshold)
            
            if np.any(binary):
                # Get bounding box
                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # Extract character with some padding
                pad = 5
                y_min = max(0, y_min - pad)
                y_max = min(h, y_max + pad + 1)
                x_min = max(0, x_min - pad)
                x_max = min(w, x_max + pad + 1)
                
                char_crop = imo[y_min:y_max, x_min:x_max]
                crop_h, crop_w = char_crop.shape[:2]
            else:
                # No visible pixels, use center crop
                char_crop = imo
                crop_h, crop_w = h, w
            
            # Scale to fit in canvas while preserving aspect ratio
            scale = min((self.canvas_size - 20) / crop_h, 
                       (self.canvas_size - 20) / crop_w)
            
            if scale < 1:
                new_h = int(crop_h * scale)
                new_w = int(crop_w * scale)
                char_crop = cv2.resize(char_crop, (new_w, new_h), 
                                     interpolation=cv2.INTER_AREA)
                crop_h, crop_w = new_h, new_w
            elif scale > 1.5:
                # Character is too small, use original method
                scale = 1
                char_crop = imo
                crop_h, crop_w = h, w
            
            # Center on canvas
            y = (self.canvas_size - crop_h) // 2
            x = (self.canvas_size - crop_w) // 2
            
            # Make sure we're within bounds
            y = max(0, min(y, self.canvas_size - crop_h))
            x = max(0, min(x, self.canvas_size - crop_w))
            
            # Place character on background - FIXED SLICE
            bg[y:y + crop_h, x:x + crop_w] = char_crop
            
            # Convert to PIL
            pil_im = Image.fromarray(bg.astype('uint8')).convert('RGB')
            
            # Debug: Save final image
            if debug:
                pil_im.save(f"debug_final_{char}.png")
                
                # Also save visualization of placement
                debug_img = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
                debug_img[:, :, 0] = bg  # Red channel = character
                debug_img[y:y+crop_h, x:x+crop_w, 1] = 100  # Green box = placement
                debug_img[y:y+crop_h, x:x+crop_w, 2] = 0
                Image.fromarray(debug_img).save(f"debug_placement_{char}.png")
            
            # Safety check: Ensure character is visible
            if np.max(bg) - np.min(bg) < 10:  # Mostly uniform
                print(f"  Warning: Character '{char}' appears blank")
                # Return a default X shape for invalid characters
                return self._create_default_x_marker(char)
            
            return pil_im
            
        except Exception as e:
            print(f"  Error rendering character '{char}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_default_x_marker(self, char: str) -> Image.Image:
        """Create a default X marker for invalid/unrenderable characters"""
        img = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)
        
        # Draw a red X
        cv2.line(img, (50, 50), (self.canvas_size-50, self.canvas_size-50), 
                (255, 0, 0), 5)
        cv2.line(img, (self.canvas_size-50, 50), (50, self.canvas_size-50), 
                (255, 0, 0), 5)
        
        # Add character label
        cv2.putText(img, char, (self.canvas_size//2 - 20, self.canvas_size//2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return Image.fromarray(img)
    
    def render_with_safety_check(self, font, char: str) -> Optional[Image.Image]:
        """
        Render character with additional safety checks for edge cases
        """
        result = self.ttf2im_fixed(font, char, debug=False)
        
        if result is None:
            return None
        
        # Convert to numpy for analysis
        img_array = np.array(result.convert('L'))  # Convert to grayscale
        
        # Check for edge artifacts (character touching edges)
        edge_thickness = 2
        top_edge = img_array[:edge_thickness, :]
        bottom_edge = img_array[-edge_thickness:, :]
        left_edge = img_array[:, :edge_thickness]
        right_edge = img_array[:, -edge_thickness:]
        
        # Check if character is too close to edges
        edge_threshold = 200  # Below this value indicates character pixel
        edges_touching = [
            np.any(top_edge < edge_threshold),
            np.any(bottom_edge < edge_threshold),
            np.any(left_edge < edge_threshold),
            np.any(right_edge < edge_threshold)
        ]
        
        if any(edges_touching):
            # Character is touching edges, render with more padding
            print(f"  Warning: Character '{char}' touches edges, adjusting...")
            # Re-render with larger canvas
            original_size = self.canvas_size
            self.canvas_size = int(original_size * 1.5)
            result = self.ttf2im_fixed(font, char, debug=False)
            self.canvas_size = original_size
            
            # Resize back to original size
            if result:
                result = result.resize((original_size, original_size), Image.LANCZOS)
        
        return result


class FontManager:
    """
    Efficient font manager with robust rendering and Unicode character support checking
    """
    
    def __init__(self, font_paths: List[str], font_size: int = 256, canvas_size: int = 256):
        """
        Initialize font manager with multiple font paths
        
        Args:
            font_paths: List of paths to TTF font files
            font_size: Default font size for rendering
            canvas_size: Size of output image
        """
        self.font_paths = font_paths
        self.font_size = font_size
        self.canvas_size = canvas_size
        
        self.available_fonts = []
        self.unicode_coverage_cache = {}
        self.renderer = FontRenderer(font_size, canvas_size)
        
        self._load_fonts()
    
    def _load_fonts(self):
        """Load and validate all fonts"""
        print(f"\nLoading {len(self.font_paths)} font(s)...")
        
        for font_path in self.font_paths:
            try:
                if not os.path.exists(font_path):
                    print(f"  ✗ Font file not found: {font_path}")
                    continue
                
                # Load with pygame for rendering
                font = pygame.freetype.Font(font_path, size=self.font_size)
                
                # Load with fontTools for Unicode coverage checking
                ttfont = TTFont(font_path)
                
                # Get font name for display
                try:
                    name_record = ttfont['name'].getName(4, 3, 1)  # Full font name
                    font_name = name_record.toUnicode() if name_record else Path(font_path).stem
                except:
                    font_name = Path(font_path).stem
                
                # Test font rendering with a sample character
                test_char = "A" if self._has_character(ttfont, ord("A")) else next(iter(self._get_unicode_coverage(ttfont)), "?")
                
                if test_char:
                    test_image = self.renderer.ttf2im_fixed(font, test_char, debug=False)
                    if test_image is None:
                        print(f"  ✗ Font '{font_name}' failed rendering test")
                        continue
                
                self.available_fonts.append({
                    'path': font_path,
                    'name': font_name,
                    'pygame_font': font,
                    'ttfont': ttfont,
                    'unicode_coverage': self._get_unicode_coverage(ttfont)
                })
                
                print(f"  ✓ Loaded: {font_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to load font {font_path}: {e}")
        
        if not self.available_fonts:
            raise ValueError("No valid fonts were loaded")
        
        print(f"Successfully loaded {len(self.available_fonts)} font(s)")
    
    @lru_cache(maxsize=128)
    def _get_unicode_coverage(self, ttfont: TTFont) -> Set[int]:
        """Extract all supported Unicode code points from a font"""
        coverage = set()
        if 'cmap' in ttfont:
            cmap = ttfont['cmap']
            for subtable in cmap.tables:
                coverage.update(subtable.cmap.keys())
        return coverage
    
    def _has_character(self, ttfont: TTFont, char_code: int) -> bool:
        """Check if a font has a specific character"""
        if 'cmap' not in ttfont:
            return False
        
        cmap = ttfont['cmap']
        for subtable in cmap.tables:
            if char_code in subtable.cmap:
                return True
        return False
    
    @lru_cache(maxsize=1024)
    def get_supporting_font(self, char: str) -> Optional[Dict]:
        """
        Find the first font that supports the given character
        
        Args:
            char: Single character string
            
        Returns:
            Font dictionary or None if no font supports the character
        """
        char_code = ord(char)
        
        for font_info in self.available_fonts:
            if char_code in font_info['unicode_coverage']:
                return font_info
        
        return None
    
    def can_render_character(self, char: str) -> bool:
        """Check if any loaded font can render the character"""
        return self.get_supporting_font(char) is not None
    
    @lru_cache(maxsize=1024)
    def render_character(self, char: str) -> Optional[Image.Image]:
        """
        Render character using the first available supporting font
        
        Args:
            char: Single character string
            
        Returns:
            PIL Image or None if character cannot be rendered
        """
        font_info = self.get_supporting_font(char)
        if not font_info:
            print(f"    No font supports character: '{char}' (U+{ord(char):04X})")
            return None
        
        try:
            # Render with safety checks
            result = self.renderer.render_with_safety_check(font_info['pygame_font'], char)
            
            if result is None:
                print(f"    Failed to render character: '{char}'")
                return None
            
            # Additional quality check
            img_array = np.array(result.convert('L'))
            
            # Check if image is mostly empty
            if np.std(img_array) < 5:  # Very low contrast
                print(f"    Warning: Character '{char}' rendered as mostly blank")
                # Try alternative rendering approach
                result = self._render_alternative(char, font_info['pygame_font'])
            
            return result
            
        except Exception as e:
            print(f"    Error rendering character '{char}': {e}")
            return None
    
    def _render_alternative(self, char: str, font) -> Image.Image:
        """Alternative rendering method for difficult characters"""
        # Create larger surface
        surface_size = self.canvas_size * 2
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 255))  # White background
        
        # Render character
        text_surface, text_rect = font.render(
            char,
            fgcolor=(0, 0, 0),  # Black
            size=self.font_size * 2  # Larger size
        )
        
        # Center
        text_rect.center = (surface_size // 2, surface_size // 2)
        surface.blit(text_surface, text_rect)
        
        # Convert to PIL and resize
        img_str = pygame.image.tostring(surface, "RGBA")
        pil_image = Image.frombytes("RGBA", (surface_size, surface_size), img_str)
        
        # Convert to RGB and resize
        pil_image = pil_image.convert("RGB").resize(
            (self.canvas_size, self.canvas_size), 
            Image.LANCZOS
        )
        
        return pil_image
    
    def batch_render_characters(self, characters: List[str]) -> Dict[str, Optional[Image.Image]]:
        """
        Render multiple characters efficiently
        
        Returns:
            Dictionary mapping character to rendered image
        """
        results = {}
        
        for char in characters:
            results[char] = self.render_character(char)
        
        return results
    
    def get_font_statistics(self) -> Dict:
        """Get statistics about loaded fonts"""
        stats = {
            'total_fonts': len(self.available_fonts),
            'font_names': [f['name'] for f in self.available_fonts],
            'font_paths': [f['path'] for f in self.available_fonts],
            'total_glyphs_by_font': {},
            'sample_coverage': {}
        }
        
        # Test coverage with common characters
        test_sets = {
            'basic_latin': "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            'common_chinese': "的一是不了人我在有他这为之大来以个中上们",
            'punctuation': "，。？！；：""''（）【】《》"
        }
        
        for font_info in self.available_fonts:
            font_name = font_info['name']
            coverage = font_info['unicode_coverage']
            
            stats['total_glyphs_by_font'][font_name] = len(coverage)
            
            # Calculate coverage for each test set
            coverage_stats = {}
            for set_name, test_chars in test_sets.items():
                supported = sum(1 for c in test_chars if ord(c) in coverage)
                total = len(test_chars)
                coverage_pct = (supported / total * 100) if total > 0 else 0
                coverage_stats[set_name] = {
                    'supported': supported,
                    'total': total,
                    'percentage': round(coverage_pct, 1)
                }
            
            stats['sample_coverage'][font_name] = coverage_stats
        
        return stats


class FontDiffuserBatchProcessor:
    """Process Excel file and generate fonts for similar characters with robust rendering"""
    
    def __init__(self, args, pipe, font_manager: FontManager):
        self.args = args
        self.pipe = pipe
        self.font_manager = font_manager
        self.device = args.device
        
        # Create transforms
        self.content_transforms, self.style_transforms = self._create_transforms()
        
        # Statistics tracking
        self.stats = {
            'characters_processed': 0,
            'characters_skipped_no_font': 0,
            'characters_skipped_render_failed': 0,
            'generation_errors': 0,
            'fonts_used': defaultdict(int),
            'edge_cases_fixed': 0
        }
    
    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create image transforms for content and style images"""
        content_transforms = transforms.Compose([
            transforms.Resize(
                self.args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        style_transforms = transforms.Compose([
            transforms.Resize(
                self.args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return content_transforms, style_transforms
    
    def prepare_images(self, char: str, style_image_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Image.Image]]:
        """
        Prepare content and style images for a character with robust error handling
        """
        try:
            # Load style image
            if not os.path.exists(style_image_path):
                raise FileNotFoundError(f"Style image not found: {style_image_path}")
            
            style_image = Image.open(style_image_path).convert('RGB')
            style_image_tensor = self.style_transforms(style_image)[None, :].to(self.device)
            
            # Prepare content image using font manager
            if self.args.character_input:
                # Check font support
                if not self.font_manager.can_render_character(char):
                    print(f"    Character '{char}' (U+{ord(char):04X}) not supported by any font")
                    self.stats['characters_skipped_no_font'] += 1
                    return None, None, None
                
                # Render character
                content_image = self.font_manager.render_character(char)
                if content_image is None:
                    print(f"    Failed to render character '{char}'")
                    self.stats['characters_skipped_render_failed'] += 1
                    return None, None, None
                
                # Check rendering quality
                img_array = np.array(content_image.convert('L'))
                
                # Detect potential rendering issues
                issues = self._detect_rendering_issues(img_array, char)
                
                if issues:
                    print(f"    Rendering issues for '{char}': {', '.join(issues)}")
                    
                    # Try to fix common issues
                    if 'edge_touching' in issues:
                        content_image = self._fix_edge_touching(content_image)
                        self.stats['edge_cases_fixed'] += 1
                
                content_image_pil = content_image.copy()
                
                # Track which font was used
                font_info = self.font_manager.get_supporting_font(char)
                if font_info:
                    self.stats['fonts_used'][font_info['name']] += 1
            else:
                # If not using character input, use white image
                content_image = Image.new('RGB', (256, 256), color='white')
                content_image_pil = None
            
            # Convert content image to tensor
            content_image_tensor = self.content_transforms(content_image)[None, :].to(self.device)
            
            # Debug: Save the prepared content image
            if hasattr(self.args, 'debug') and self.args.debug:
                debug_dir = Path(self.args.output_base_dir) / "debug" / "content_images"
                debug_dir.mkdir(parents=True, exist_ok=True)
                content_image.save(debug_dir / f"content_{char}.png")
            
            return content_image_tensor, style_image_tensor, content_image_pil
            
        except Exception as e:
            print(f"    Error preparing images for '{char}': {e}")
            self.stats['generation_errors'] += 1
            return None, None, None
    
    def _detect_rendering_issues(self, img_array: np.ndarray, char: str) -> List[str]:
        """Detect potential rendering issues in character image"""
        issues = []
        
        # Check if image is mostly one color
        if np.std(img_array) < 10:
            issues.append('low_contrast')
        
        # Check if character touches edges
        edge_thickness = 3
        edges = [
            img_array[:edge_thickness, :],  # Top
            img_array[-edge_thickness:, :],  # Bottom
            img_array[:, :edge_thickness],   # Left
            img_array[:, -edge_thickness:]   # Right
        ]
        
        edge_threshold = 200
        for i, edge in enumerate(edges):
            if np.any(edge < edge_threshold):
                issues.append('edge_touching')
                break
        
        # Check if character is too small
        binary = img_array < 240  # Threshold for character pixels
        character_pixels = np.sum(binary)
        total_pixels = img_array.size
        
        if character_pixels < total_pixels * 0.01:  # Less than 1% of pixels
            issues.append('too_small')
        
        return issues
    
    def _fix_edge_touching(self, image: Image.Image) -> Image.Image:
        """Fix edge-touching by adding padding"""
        # Convert to numpy
        img_array = np.array(image.convert('RGB'))
        
        # Add white border
        border_size = 20
        h, w = img_array.shape[:2]
        new_h, new_w = h + border_size * 2, w + border_size * 2
        
        new_array = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        new_array[border_size:border_size+h, border_size:border_size+w] = img_array
        
        return Image.fromarray(new_array).resize((h, w), Image.LANCZOS)
    
    def generate_character(self, char: str, style_image_path: str) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
        """Generate a single character using FontDiffuser"""
        # Set seed for reproducibility if specified
        if self.args.seed:
            set_seed(seed=self.args.seed)
        
        # Prepare images
        content_tensor, style_tensor, content_pil = self.prepare_images(char, style_image_path)
        
        if content_tensor is None or style_tensor is None:
            return None, None
        
        try:
            # Generate using FontDiffuser pipeline
            with torch.no_grad():
                images = self.pipe.generate(
                    content_images=content_tensor,
                    style_images=style_tensor,
                    batch_size=1,
                    order=self.args.order,
                    num_inference_step=self.args.num_inference_steps,
                    content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                    t_start=self.args.t_start,
                    t_end=self.args.t_end,
                    dm_size=self.args.content_image_size,
                    algorithm_type=self.args.algorithm_type,
                    skip_type=self.args.skip_type,
                    method=self.args.method,
                    correcting_x0_fn=self.args.correcting_x0_fn
                )
            
            self.stats['characters_processed'] += 1
            return images[0], content_pil
            
        except Exception as e:
            print(f"    Error generating character '{char}': {e}")
            self.stats['generation_errors'] += 1
            return None, None

def parse_excel_batch_args():
    """Parse arguments for batch processing with multi-font support"""
    from configs.fontdiffuser import get_parser
    
    parser = get_parser()
    
    # Existing arguments
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--character_input", action="store_true", default=True)
    parser.add_argument("--style_image_path", type=str, required=True, 
                       help="Path to style reference image")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Font arguments
    parser.add_argument("--ttf_path", type=str, action='append', 
                       help="Path to TTF font file (can be specified multiple times)")
    parser.add_argument("--font_dir", type=str, default=None,
                       help="Directory containing multiple TTF fonts (all .ttf files will be loaded)")
    
    # Batch processing arguments
    parser.add_argument("--excel_file", type=str, required=True,
                       help="Path to Excel file with character data")
    parser.add_argument("--output_base_dir", type=str, default="./fontdiffuser_batch_output",
                       help="Base directory for all outputs")
    parser.add_argument("--skip_input_char", action="store_true",
                       help="Skip generating the input character")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Maximum number of rows to process")
    parser.add_argument("--debug", action="store_true",
                   help="Save debug images and additional information")

    
    args = parser.parse_args()
    
    # Set image sizes
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    
    # Collect font paths
    font_paths = []
    
    # Add individual font paths
    if args.ttf_path:
        font_paths.extend([p for p in args.ttf_path if p])
    
    # Add fonts from directory
    if args.font_dir and os.path.exists(args.font_dir):
        font_paths.extend([
            str(p) for p in Path(args.font_dir).glob("*.ttf")
        ])
        font_paths.extend([
            str(p) for p in Path(args.font_dir).glob("*.TTF")
        ])
    
    # Remove duplicates
    font_paths = list(dict.fromkeys(font_paths))
    
    if not font_paths:
        # Try default location
        default_font = Path(__file__).parent / "fonts" / "default.ttf"
        if default_font.exists():
            font_paths = [str(default_font)]
        else:
            raise ValueError("No font files specified or found. Use --ttf_path or --font_dir")
    
    args.font_paths = font_paths
    
    return args


def load_fontdiffuser_pipeline(args):
    """Load the FontDiffuser pipeline once"""
    print("Loading FontDiffuser model...")
    
    # Load model components
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    
    # Create model
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    model.to(args.device)
    
    # Load scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    
    # Create pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    
    print("✓ FontDiffuser pipeline loaded successfully")
    return pipe


def main_batch_processing():
    """Main function for batch processing Excel file with multi-font support"""
    args = parse_excel_batch_args()
    
    print(f"\n{'='*70}")
    print(f"FONTDIFFUSER BATCH PROCESSING")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = Path(args.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_args_to_yaml(
        args=args, 
        output_file=str(output_dir / "batch_config.yaml")
    )
    
    # Initialize font manager with multiple fonts
    print(f"\nLoading fonts...")
    font_manager = FontManager(
        font_paths=args.font_paths,
        font_size=128  # You can make this configurable if needed
    )
    
    # Load pipeline once
    print(f"\nLoading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline(args)
    
    # Create processor
    processor = FontDiffuserBatchProcessor(args, pipe, font_manager)
    
    # Process Excel file
    print(f"\nProcessing Excel file: {args.excel_file}")
    results = processor.process_excel_file(
        excel_path=args.excel_file,
        base_output_dir=str(output_dir),
        style_image_path=args.style_image_path,
        generate_input_char=not args.skip_input_char,
        max_rows=args.max_rows
    )
    
    return results


if __name__ == "__main__":
    try:
        results = main_batch_processing()
        print(f"\n✓ Batch processing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Example
python sample_excel.py \
    --excel_file "characters.xlsx" \
    --style_image_path "./style/A.png" \
    --ckpt_dir "./checkpoints" \
    --ttf_path "./fonts/KaiXinSongA.ttf" \
    --output_base_dir "./output" \
    --debug  # Enable debug mode
"""

"""output/
├── batch_config.yaml
├── font_statistics.json
├── global_summary.json
├── debug/                          # Debug information
│   ├── content_images/            # All prepared content images
│   │   ├── content_A.png
│   │   ├── content_B.png
│   │   └── ...
│   └── rendering_issues.json      # Log of rendering issues
├── row_001_𠀖/
└── ...
"""
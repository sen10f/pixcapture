#!/usr/bin/env python3
"""
Video to DMX Art-Net Converter with Audio Support and Master Channel
Convert video pixel data to DMX with master brightness channel and send via Art-Net

Required packages:
pip install dearpygui opencv-python numpy pygame

"""

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import socket
import struct
import threading
import time
import os
import functools
from typing import Optional, Tuple, List
import pygame
import subprocess
import json

class AudioController:
    """Audio playback controller with device selection"""
    
    def __init__(self):
        try:
            pygame.mixer.pre_init()
            pygame.mixer.init()
        except:
            print("Audio initialization failed")
        self.sound = None
        self.is_playing = False
        self.is_paused = False
        self.start_time = 0
        self.pause_time = 0
        self.current_pos = 0
        self.available_devices = []
        self.selected_device = None
        self.volume = 0.7
        self.audio_thread = None
        self.stop_audio = False
        self._get_audio_devices()
        
    def _get_audio_devices(self):
        """Get available audio output devices"""
        try:
            # For Windows - use wmic to get audio devices
            if os.name == 'nt':
                try:
                    result = subprocess.run(
                        ['wmic', 'sounddev', 'get', 'name', '/format:csv'],
                        capture_output=True, text=True, timeout=5
                    )
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    devices = []
                    for line in lines:
                        if line.strip() and ',' in line:
                            name = line.split(',')[-1].strip()
                            if name and name != 'Name':
                                devices.append(name)
                    self.available_devices = devices[:10]  # Limit to 10 devices
                except:
                    self.available_devices = ["Default", "Speakers", "Headphones"]
            else:
                # For Linux/Mac - basic device list
                self.available_devices = ["Default", "Speakers", "Headphones", "HDMI Audio"]
        except:
            self.available_devices = ["Default"]
        
        if not self.available_devices:
            self.available_devices = ["Default"]
    
    def load_audio(self, video_path: str) -> bool:
        """Extract and load audio from video file"""
        try:
            # Extract audio using ffmpeg (if available)
            audio_path = "temp_audio.wav"
            
            # Try to extract audio with ffmpeg
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                    '-ar', '44100', '-ac', '2', '-y', audio_path
                ], check=True, capture_output=True)
                
                # Load the extracted audio
                self.sound = pygame.mixer.Sound(audio_path)
                return True
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg not available or failed, try direct loading
                try:
                    # Try loading video file directly (some formats work)
                    self.sound = pygame.mixer.Sound(video_path)
                    return True
                except:
                    print("Could not extract or load audio from video file")
                    return False
                    
        except Exception as e:
            print(f"Audio loading error: {e}")
            return False
    
    def play(self, start_pos: float = 0):
        """Play audio from specified position"""
        if self.sound:
            try:
                # Stop any current playback
                pygame.mixer.stop()
                
                # Play the sound
                pygame.mixer.Sound.play(self.sound)
                pygame.mixer.music.set_volume(self.volume)
                
                self.is_playing = True
                self.is_paused = False
                self.start_time = time.time() - start_pos
                return True
            except Exception as e:
                print(f"Audio play error: {e}")
                return False
        return False
    
    def stop(self):
        """Stop audio playback"""
        try:
            pygame.mixer.stop()
            self.is_playing = False
            self.is_paused = False
            self.current_pos = 0
        except Exception as e:
            print(f"Audio stop error: {e}")
    
    def pause(self):
        """Pause audio playback"""
        try:
            pygame.mixer.pause()
            self.is_paused = True
            self.pause_time = time.time()
        except Exception as e:
            print(f"Audio pause error: {e}")
    
    def resume(self):
        """Resume audio playback"""
        try:
            pygame.mixer.unpause()
            self.is_paused = False
            # Adjust start time to account for pause duration
            if self.pause_time > 0:
                pause_duration = time.time() - self.pause_time
                self.start_time += pause_duration
        except Exception as e:
            print(f"Audio resume error: {e}")
    
    def set_volume(self, volume: float):
        """Set audio volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(self.volume)
        except:
            pass
    
    def get_position(self) -> float:
        """Get current playback position in seconds"""
        if self.is_playing and not self.is_paused:
            return time.time() - self.start_time
        return self.current_pos
    
    def is_audio_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self.is_playing and pygame.mixer.get_busy()

class ArtNetController:
    """Art-Net protocol DMX data sender"""
    
    def __init__(self, target_ip: str = "255.255.255.255", port: int = 6454):
        self.target_ip = target_ip
        self.port = port
        self.socket = None
        self.universe = 0
        self.sequence = 0
        self.is_connected = False
        self.debug_mode = False
        self.packet_count = 0
        self._init_socket()
        
    def _init_socket(self):
        """Initialize UDP socket for Art-Net"""
        try:
            if self.socket:
                self.socket.close()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.settimeout(1.0)  # 1 second timeout
            self.is_connected = True
            print(f"Art-Net socket initialized for {self.target_ip}:{self.port}")
        except Exception as e:
            print(f"Failed to initialize Art-Net socket: {e}")
            self.is_connected = False
    
    def enable_debug(self, enabled: bool = True):
        """Enable/disable debug output"""
        self.debug_mode = enabled
    
    def _debug_packet(self, packet: bytearray):
        """Print packet contents in hex format for debugging"""
        if not self.debug_mode:
            return
        
        print(f"\n--- Art-Net Packet #{self.packet_count} ---")
        print(f"Target: {self.target_ip}:{self.port}")
        print(f"Size: {len(packet)} bytes")
        
        # Header breakdown
        print(f"ID: {packet[0:8]} ({packet[0:7].decode('ascii', errors='ignore')})")
        print(f"OpCode: 0x{struct.unpack('<H', packet[8:10])[0]:04x}")
        print(f"ProtVer: {struct.unpack('>H', packet[10:12])[0]}")
        print(f"Sequence: {packet[12]}")
        print(f"Physical: {packet[13]}")
        print(f"SubUni: {packet[14]} (Universe: {packet[14] | (packet[15] << 4)})")
        print(f"Net: {packet[15]}")
        print(f"Length: {struct.unpack('>H', packet[16:18])[0]} (should be 512 for DMX)")
        
        # Data analysis
        data_start = 18
        dmx_data = packet[data_start:]
        
        # Find the highest non-zero channel
        last_nonzero = 0
        for i in range(len(dmx_data) - 1, -1, -1):
            if dmx_data[i] != 0:
                last_nonzero = i + 1
                break
        
        print(f"DMX Data: 512 channels total")
        print(f"Active channels: 1-{last_nonzero} (last non-zero at channel {last_nonzero})")
        
        # Data preview (first 16 bytes)
        data_preview = dmx_data[0:16]
        hex_str = ' '.join([f'{b:02x}' for b in data_preview])
        print(f"Data (first 16 bytes): {hex_str}")
        
        # RGB+Master preview for first few pixels
        rgbm_pixels = []
        for i in range(0, min(16, len(dmx_data)), 4):
            if i + 3 < len(dmx_data):
                r, g, b, m = dmx_data[i], dmx_data[i+1], dmx_data[i+2], dmx_data[i+3]
                rgbm_pixels.append(f"Pixel{i//4+1}:({r},{g},{b},M{m})")
        
        if rgbm_pixels:
            print(f"RGBM Preview: {' '.join(rgbm_pixels[:4])}")  # Show first 4 pixels
        
        # Channel utilization
        zero_channels = dmx_data.count(0)
        used_channels = 512 - zero_channels
        utilization = (used_channels / 512) * 100
        print(f"Channel utilization: {used_channels}/512 ({utilization:.1f}%)")
        
        print("--- End Packet ---\n")
        
    def send_dmx(self, dmx_data: List[int], universe: int = 0):
        """Send DMX data as Art-Net packet"""
        if not self.is_connected or not self.socket:
            self._init_socket()
            if not self.is_connected:
                return False

        try:
            # Ensure DMX data is integers between 0-255
            original_length = len(dmx_data)
            dmx_data = [max(0, min(255, int(val))) for val in dmx_data]

            # Pad or truncate DMX data to exactly 512 bytes
            if len(dmx_data) > 512:
                dmx_data = dmx_data[:512]
                if self.debug_mode:
                    print(f"Warning: DMX data truncated from {original_length} to 512 channels")
            elif len(dmx_data) < 512:
                padding_needed = 512 - len(dmx_data)
                dmx_data.extend([0] * padding_needed)
                if self.debug_mode:
                    print(f"Info: DMX data padded from {original_length} to 512 channels")

            # Build Art-Net packet
            packet = bytearray()
            packet.extend(b"Art-Net\x00")                   # ID
            packet.extend(struct.pack('<H', 0x5000))        # OpCode (OpDmx)
            packet.extend(struct.pack('>H', 14))            # ProtVer (14)

            # Sequence number: ensure it's 1-255
            if self.sequence == 0:
                self.sequence = 1
            packet.append(self.sequence)                    # Sequence
            packet.append(0)                                # Physical

            # Universe encoding (Art-Net 4): SubUni = universe & 0xFF, Net = (universe >> 8) & 0x7F
            sub_uni = universe & 0xFF
            net = (universe >> 8) & 0x7F
            packet.append(sub_uni)
            packet.append(net)

            packet.extend(struct.pack('>H', 512))           # Length (512 bytes)
            packet.extend(dmx_data)                         # DMX Data (512 bytes)

            if self.debug_mode:
                self._debug_packet(packet)

            bytes_sent = self.socket.sendto(packet, (self.target_ip, self.port))
            self.packet_count += 1

            # Update sequence number (skip 0)
            self.sequence = (self.sequence + 1) % 256
            if self.sequence == 0:
                self.sequence = 1

            expected_size = 18 + 512
            if bytes_sent == expected_size:
                if self.debug_mode:
                    print(f"✓ Art-Net packet sent successfully: {bytes_sent} bytes")
                return True
            else:
                print(f"✗ Art-Net packet size mismatch: sent {bytes_sent}, expected {expected_size}")
                return False

        except Exception as e:
            print(f"Art-Net send error: {e}")
            self.is_connected = False
            return False

    
    def send_artpoll(self):
        """Send Art-Poll packet to discover Art-Net nodes"""
        try:
            packet = bytearray()
            packet.extend(b"Art-Net\x00")  # ID
            packet.extend(struct.pack('<H', 0x2000))  # OpCode - ArtPoll
            packet.extend(struct.pack('>H', 14))  # ProtVer
            packet.append(0x02)  # TalkToMe - send ArtPollReply
            packet.append(0x00)  # Priority
            
            self.socket.sendto(packet, (self.target_ip, self.port))
            print("Art-Poll packet sent")
            return True
        except Exception as e:
            print(f"Art-Poll send error: {e}")
            return False
    
    def get_stats(self):
        """Get transmission statistics"""
        return {
            'packets_sent': self.packet_count,
            'target_ip': self.target_ip,
            'port': self.port,
            'universe': self.universe,
            'sequence': self.sequence,
            'connected': self.is_connected
        }

class VideoProcessor:
    """Video processing and pixel data extraction"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30
        self.current_frame = 0
        self.is_playing = False
        self.frame_data = None
        self.duration = 0
        
    def load_video(self, filepath: str) -> bool:
        """Load video file"""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(filepath)
            if not self.cap.isOpened():
                return False
                
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            self.current_frame = 0
            
            print(f"Video loaded: {self.frame_count} frames, {self.fps} FPS, {self.duration:.2f}s")
            return True
        except Exception as e:
            print(f"Video loading error: {e}")
            return False
    
    def get_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        """Get specified frame or next frame"""
        if not self.cap:
            return None
            
        try:
            if frame_number is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.current_frame = frame_number
                
            ret, frame = self.cap.read()
            if ret:
                self.frame_data = frame
                return frame
        except Exception as e:
            print(f"Frame read error: {e}")
        return None
    
    def seek_to_time(self, time_seconds: float):
        """Seek to specific time in video"""
        if not self.cap:
            return False
        
        try:
            frame_number = int(time_seconds * self.fps)
            frame_number = max(0, min(frame_number, self.frame_count - 1))
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        except Exception as e:
            print(f"Seek error: {e}")
            return False
    
    def get_current_time(self) -> float:
        """Get current playback time in seconds"""
        if self.fps > 0:
            return self.current_frame / self.fps
        return 0
    
    def extract_pixel_data(self, frame: np.ndarray, grid_size: Tuple[int, int] = (16, 16), include_master: bool = True) -> List[List[int]]:
        """Extract pixel data from frame and convert to RGB with optional master channel"""
        try:
            height, width = frame.shape[:2]
            grid_w, grid_h = grid_size
            
            # Enable debug only for first few frames or when manually requested
            debug_enabled = hasattr(self, '_debug_pixel_extraction') and self._debug_pixel_extraction
            
            if debug_enabled:
                print(f"Extracting pixel data: frame={width}x{height}, grid={grid_w}x{grid_h}, master={include_master}")
            
            # Sample pixels based on grid size
            pixel_data = []
            
            for y in range(grid_h):
                for x in range(grid_w):
                    # Calculate grid position - center of each grid cell
                    pixel_x = int((x + 0.5) * width / grid_w)
                    pixel_y = int((y + 0.5) * height / grid_h)
                    
                    # Ensure coordinates are within bounds
                    pixel_x = max(0, min(pixel_x, width - 1))
                    pixel_y = max(0, min(pixel_y, height - 1))
                    
                    # BGR -> RGB conversion (OpenCV uses BGR format)
                    b, g, r = frame[pixel_y, pixel_x]
                    
                    if include_master:
                        # Calculate master value using luminance formula
                        master = int(0.299 * r + 0.587 * g + 0.114 * b)
                        pixel_data.append([int(r), int(g), int(b), master])
                    else:
                        pixel_data.append([int(r), int(g), int(b)])
                    
                    # Debug first few pixels only
                    if debug_enabled and len(pixel_data) <= 3:
                        if include_master:
                            print(f"Pixel ({x},{y}) -> frame({pixel_x},{pixel_y}) = RGB({r},{g},{b}) Master({master})")
                        else:
                            print(f"Pixel ({x},{y}) -> frame({pixel_x},{pixel_y}) = RGB({r},{g},{b})")
                    
            if debug_enabled:
                channels_per_pixel = 4 if include_master else 3
                print(f"Extracted {len(pixel_data)} pixels ({channels_per_pixel} channels each)")
                
            return pixel_data
        except Exception as e:
            print(f"Pixel extraction error: {e}")
            return []

class VideoToDMXApp:
    """Main application class"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.artnet_controller = ArtNetController()
        self.audio_controller = AudioController()
        self.playback_thread = None
        self.stop_playback = False
        
        # Settings
        self.grid_width = 16
        self.grid_height = 16
        self.brightness = 255
        self.fps_override = 30
        self.current_file = ""
        self.current_dmx_data = []
        self.loop_playback = False
        self.audio_enabled = True
        self.master_enabled = True  # New: Master channel enable/disable
        
        # Zoom and pan settings
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Pixel mapping settings - updated for master channel
        self.pixel_mappings = {}  # {(x, y): {'r': channel, 'g': channel, 'b': channel, 'm': channel}}
        self.selected_pixel = None
        self.mapping_mode = False
        
        # Fixed channels settings - NEW
        self.fixed_channels = {}  # {channel: {'value': int, 'name': str, 'enabled': bool}}
        self.selected_fixed_channel = None
        
        # Window visibility flags
        self.windows = {
            "video_preview": False,
            "dmx_data": False,
            "settings": False,
            "pixel_mapping": False,
            "audio_settings": False
        }
        
        # GUI initialization
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI"""
        dpg.create_context()
        
        # Create texture first
        with dpg.texture_registry():
            # Initialize texture with dummy image
            dummy_data = np.zeros((240, 320, 4), dtype=np.float32)
            dpg.add_raw_texture(width=320, height=240, default_value=dummy_data, 
                              format=dpg.mvFormat_Float_rgba, tag="video_texture")
        
        # File selection dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.file_selected,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension("Video Files{.mp4,.avi,.mov,.mkv}", color=(255, 255, 0, 255))
            dpg.add_file_extension(".*")
        
        # Create all windows
        self.create_main_window()
        self.create_video_preview_window()
        self.create_dmx_data_window()
        self.create_settings_window()
        self.create_pixel_mapping_window()
        self.create_audio_settings_window()
        
        dpg.create_viewport(title="pixcap - 0.3.1", width=900, height=700)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        # Initialize tables
        self.update_fixed_channels_table()
        
    def create_main_window(self):
        """Create main control window"""
        with dpg.window(label="Main Control", tag="main_window", width=450, height=450):
            
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="Windows"):
                    dpg.add_menu_item(label="Video Preview", callback=self.toggle_video_preview)
                    dpg.add_menu_item(label="DMX Data", callback=self.toggle_dmx_data)
                    dpg.add_menu_item(label="Settings", callback=self.toggle_settings)
                    dpg.add_menu_item(label="Pixel Mapping", callback=self.toggle_pixel_mapping)
                    dpg.add_menu_item(label="Audio Settings", callback=self.toggle_audio_settings)
                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="About", callback=self.show_about)
            
            # File selection
            with dpg.group(horizontal=True):
                dpg.add_button(label="Select Video File", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_text("File: Not selected", tag="file_status")
            
            dpg.add_separator()
            
            # Fixed channels settings - NEW
            with dpg.collapsing_header(label="Fixed Channels", default_open=False):
                dpg.add_text("Set fixed values for specific DMX channels")
                dpg.add_text("Useful for dimmer, strobe, and special effect channels")
                
                # Add new fixed channel
                with dpg.group():
                    dpg.add_text("Add Fixed Channel:")
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(label="Channel", tag="new_fixed_channel", width=80, min_value=1, max_value=512)
                        dpg.add_input_int(label="Value", tag="new_fixed_value", width=80, min_value=0, max_value=255)
                        dpg.add_input_text(label="Name", tag="new_fixed_name", width=120, hint="e.g., Dimmer, Strobe")
                        dpg.add_button(label="Add", callback=self.add_fixed_channel, width=50)
                
                # Fixed channels table
                dpg.add_separator()
                with dpg.table(header_row=True, tag="fixed_channels_table", height=150, scrollY=True):
                    dpg.add_table_column(label="Ch", width_fixed=True, init_width_or_weight=40)
                    dpg.add_table_column(label="Value", width_fixed=True, init_width_or_weight=50)
                    dpg.add_table_column(label="Name", width_fixed=True, init_width_or_weight=100)
                    dpg.add_table_column(label="Enabled", width_fixed=True, init_width_or_weight=60)
                    dpg.add_table_column(label="Actions", width_fixed=True, init_width_or_weight=70)
                
                # Fixed channel controls
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Enable All", callback=self.enable_all_fixed_channels, width=80)
                    dpg.add_button(label="Disable All", callback=self.disable_all_fixed_channels, width=80)
                    dpg.add_button(label="Clear All", callback=self.clear_all_fixed_channels, width=80)
                
                # Common presets
                dpg.add_separator()
                dpg.add_text("Common Presets:")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Master Dimmer (Ch1=255)", callback=lambda: self.add_preset_channel(1, 255, "Master Dimmer"), width=150)
                    dpg.add_button(label="Strobe Off (Ch5=0)", callback=lambda: self.add_preset_channel(5, 0, "Strobe Off"), width=150)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Color Wheel Reset (Ch6=0)", callback=lambda: self.add_preset_channel(6, 0, "Color Wheel"), width=150)
                    dpg.add_button(label="Pan Center (Ch7=128)", callback=lambda: self.add_preset_channel(7, 128, "Pan Center"), width=150)
            
            dpg.add_separator()
            
            # Quick settings
            with dpg.collapsing_header(label="Quick Settings", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Grid Size:")
                    dpg.add_input_int(label="W", default_value=16, tag="grid_width_main", width=60, 
                                    callback=self.update_grid_settings)
                    dpg.add_input_int(label="H", default_value=16, tag="grid_height_main", width=60,
                                    callback=self.update_grid_settings)
                
                dpg.add_slider_int(label="Brightness", default_value=255, min_value=0, max_value=255, 
                                 tag="brightness_main", callback=self.update_brightness)
                
                dpg.add_checkbox(label="Loop Playback", tag="loop_playback_main", 
                               callback=self.update_loop_setting)
                dpg.add_checkbox(label="Enable Audio", tag="audio_enabled_main", default_value=True,
                               callback=self.update_audio_setting)
                # New: Master channel setting
                dpg.add_checkbox(label="Enable Master Channel", tag="master_enabled_main", default_value=True,
                               callback=self.update_master_setting)
            
            dpg.add_separator()
            
            # Playback controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=self.play_video, tag="play_btn")
                dpg.add_button(label="Stop", callback=self.stop_video, tag="stop_btn", enabled=False)
                dpg.add_button(label="Restart", callback=self.restart_video, tag="restart_btn")
                dpg.add_button(label="Test Art-Net", callback=self.test_artnet)
            
            # Seek controls
            with dpg.group(horizontal=True):
                dpg.add_text("Seek:")
                dpg.add_slider_float(label="##seek_slider", default_value=0.0, min_value=0.0, max_value=1.0,
                                   tag="seek_slider", callback=self.seek_video, width=200)
                dpg.add_text("00:00", tag="time_display")
            
            # Progress bar
            dpg.add_progress_bar(label="Progress", tag="progress", width=-1)
            
            # Audio volume
            with dpg.group(horizontal=True):
                dpg.add_text("Volume:")
                dpg.add_slider_float(label="##volume_slider", default_value=0.7, min_value=0.0, max_value=1.0,
                                   tag="volume_slider", callback=self.update_volume, width=150)
                dpg.add_text("70%", tag="volume_display")
            
            # Status
            dpg.add_separator()
            with dpg.group():
                dpg.add_text("Status:")
                dpg.add_text("Ready", tag="status_text")
                dpg.add_text("Frame: 0/0", tag="frame_info")
                dpg.add_text("Duration: 00:00", tag="duration_info")
                dpg.add_text("DMX Channels: 0", tag="dmx_info")
                dpg.add_text("Audio: Not loaded", tag="audio_status")
                dpg.add_text("Art-Net: Disconnected", tag="artnet_status")
                dpg.add_text("Master: Enabled", tag="master_status")  # New: Master status
    
    def create_audio_settings_window(self):
        """Create audio settings window"""
        with dpg.window(label="Audio Settings", tag="audio_settings_window", 
                       width=400, height=300, show=False, pos=[460, 420]):
            
            dpg.add_text("Audio Output Device:")
            
            # Audio device selection
            device_names = self.audio_controller.available_devices
            dpg.add_combo(device_names, label="Output Device", tag="audio_device_combo", 
                         default_value=device_names[0] if device_names else "Default",
                         callback=self.change_audio_device, width=300)
            
            dpg.add_separator()
            
            # Audio settings
            with dpg.collapsing_header(label="Audio Settings", default_open=True):
                dpg.add_checkbox(label="Enable Audio Playback", tag="enable_audio", default_value=True,
                               callback=self.toggle_audio_playback)
                
                dpg.add_slider_float(label="Volume", default_value=0.7, min_value=0.0, max_value=1.0,
                                   tag="audio_volume", callback=self.set_audio_volume)
                
                dpg.add_checkbox(label="Sync Audio with Video", tag="sync_audio", default_value=True)
                
                dpg.add_slider_int(label="Audio Delay (ms)", default_value=0, min_value=-1000, max_value=1000,
                                 tag="audio_delay")
            
            dpg.add_separator()
            
            # Audio info
            with dpg.collapsing_header(label="Audio Information"):
                dpg.add_text("Status: Not loaded", tag="audio_info_status")
                dpg.add_text("Format: N/A", tag="audio_info_format")
                dpg.add_text("Channels: N/A", tag="audio_info_channels")
                dpg.add_text("Sample Rate: N/A", tag="audio_info_rate")
            
            dpg.add_separator()
            
            # Audio controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Test Audio", callback=self.test_audio)
                dpg.add_button(label="Refresh Devices", callback=self.refresh_audio_devices)
    
    def create_video_preview_window(self):
        """Create video preview window"""
        with dpg.window(label="Video Preview", tag="video_preview_window", 
                       width=500, height=600, show=False, pos=[420, 20]):
            
            dpg.add_text("Video Preview:")
            
            # Add mouse handlers to the image
            dpg.add_image("video_texture", width=320, height=240, tag="video_preview")
            
            # Zoom controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Zoom In", callback=self.zoom_in, width=70)
                dpg.add_button(label="Zoom Out", callback=self.zoom_out, width=70)
                dpg.add_button(label="Reset Zoom", callback=self.reset_zoom, width=80)
                dpg.add_button(label="Fit to Window", callback=self.fit_to_window, width=90)
            
            # Zoom info
            with dpg.group(horizontal=True):
                dpg.add_text("Zoom:")
                dpg.add_slider_float(label="##zoom_slider", default_value=1.0, min_value=0.25, max_value=16.0, 
                                   tag="zoom_slider", callback=self.set_zoom_level, width=150)
                dpg.add_text("1.0x", tag="zoom_text")
            
            # Pan controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="↑", callback=self.pan_up, width=30)
                dpg.add_button(label="↓", callback=self.pan_down, width=30)
                dpg.add_button(label="←", callback=self.pan_left, width=30)
                dpg.add_button(label="→", callback=self.pan_right, width=30)
                dpg.add_button(label="Center", callback=self.center_view, width=60)
            
            dpg.add_separator()
            
            # Preview controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Original Size", callback=self.reset_preview_size)
                dpg.add_button(label="Pixel Perfect", callback=self.set_pixel_perfect_view)
            
            dpg.add_separator()
            
            # Pixel display settings
            with dpg.collapsing_header(label="Display Settings", default_open=True):
                dpg.add_checkbox(label="Show Grid", tag="show_grid_preview", 
                               callback=self.update_grid_display)
                dpg.add_checkbox(label="Show Pixel Numbers", tag="show_pixel_numbers")
                dpg.add_checkbox(label="Highlight Selected Pixel", tag="highlight_pixel")
                
                # Grid appearance
                with dpg.group(horizontal=True):
                    dpg.add_text("Grid Color:")
                    dpg.add_color_edit(label="##grid_color", default_value=[0.5, 0.5, 0.5, 1.0], 
                                     tag="grid_color", width=100, no_alpha=True)
                
                dpg.add_slider_int(label="Grid Line Width", default_value=1, min_value=1, max_value=3, 
                                 tag="grid_line_width")
            
            dpg.add_separator()
            
            # Video information
            with dpg.collapsing_header(label="Video Info", default_open=True):
                dpg.add_text("Resolution: N/A", tag="video_resolution")
                dpg.add_text("FPS: N/A", tag="video_fps")
                dpg.add_text("Duration: N/A", tag="video_duration")
                dpg.add_text("Current Frame: 0", tag="current_frame_info")
                dpg.add_text("Scale Factor: 1.0x", tag="scale_factor")
                dpg.add_text("View Area: 0,0", tag="view_area")
            
            # Pixel info (when clicked)
            dpg.add_separator()
            with dpg.group():
                dpg.add_text("Pixel Info (click on preview):")
                dpg.add_text("Position: N/A", tag="pixel_position")
                dpg.add_text("RGB: N/A", tag="pixel_rgb")
                dpg.add_text("Master: N/A", tag="pixel_master")  # New: Master value display
                dpg.add_text("DMX Channel: N/A", tag="pixel_dmx_channel")
        
        # Add mouse handlers
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel_callback)
            dpg.add_mouse_click_handler(callback=self.mouse_click_callback)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag_callback)
    
    def create_dmx_data_window(self):
        """Create DMX data visualization window"""
        with dpg.window(label="DMX Data", tag="dmx_data_window", 
                       width=450, height=500, show=False, pos=[20, 320]):
            
            dpg.add_text("DMX Channel Data:")
            
            # DMX data table - updated for RGBM format
            with dpg.table(header_row=True, tag="dmx_table", height=300, scrollY=True):
                dpg.add_table_column(label="Ch", width_fixed=True, init_width_or_weight=50)
                dpg.add_table_column(label="Value", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="RGBM", width_fixed=True, init_width_or_weight=90)
                dpg.add_table_column(label="Preview", width_fixed=True, init_width_or_weight=80)
            
            dpg.add_separator()
            
            # Data export
            with dpg.group(horizontal=True):
                dpg.add_button(label="💾 Export JSON", callback=self.export_dmx_csv, width=110)
                dpg.add_button(label="🗑️ Clear Data", callback=self.clear_dmx_data, width=110)
            
            dpg.add_text("Total Channels: 0", tag="total_channels")
            dpg.add_text("Data Rate: 0 Hz", tag="data_rate")
            dpg.add_text("Format: RGB", tag="channel_format")  # New: Show current format
    
    def create_settings_window(self):
        """Create settings window"""
        with dpg.window(label="Settings", tag="settings_window", 
                       width=400, height=500, show=False, pos=[820, 20]):
            
            # Video settings
            with dpg.collapsing_header(label="Video Settings", default_open=True):
                dpg.add_input_float(label="FPS Override", default_value=30.0, tag="fps_override", width=100)
                dpg.add_checkbox(label="Loop Playback", tag="loop_playback")
                dpg.add_checkbox(label="Auto-start on Load", tag="auto_start")
                dpg.add_checkbox(label="Auto-restart when finished", tag="auto_restart")
            
            dpg.add_separator()
            
            # Grid settings
            with dpg.collapsing_header(label="Grid Settings", default_open=True):
                dpg.add_input_int(label="Grid Width", default_value=16, tag="grid_width", width=100)
                dpg.add_input_int(label="Grid Height", default_value=16, tag="grid_height", width=100)
                dpg.add_checkbox(label="Show Grid", tag="show_grid", callback=self.sync_grid_settings)
                
                # Grid presets
                dpg.add_text("Presets:")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="8x8", callback=lambda: self.set_grid_preset(8, 8))
                    dpg.add_button(label="16x16", callback=lambda: self.set_grid_preset(16, 16))
                    dpg.add_button(label="32x32", callback=lambda: self.set_grid_preset(32, 32))
                    dpg.add_button(label="16x8", callback=lambda: self.set_grid_preset(16, 8))
                    dpg.add_button(label="32x16", callback=lambda: self.set_grid_preset(32, 16))
            
            dpg.add_separator()
            
            # Channel settings - New section for master channel
            with dpg.collapsing_header(label="Channel Settings", default_open=True):
                dpg.add_checkbox(label="Enable Master Channel", tag="master_enabled", default_value=True,
                               callback=self.update_master_enabled)
                dpg.add_text("Channel Format:")
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(["RGB", "RGBM"], tag="channel_format_radio", default_value=1,
                                        callback=self.update_channel_format, horizontal=True)
                
                dpg.add_separator()
                dpg.add_text("Master Calculation Method:")
                dpg.add_combo(["Luminance (0.299R+0.587G+0.114B)", "Average (R+G+B)/3", "Maximum (max RGB)", "Custom"],
                             label="Master Method", tag="master_method", default_value="Luminance (0.299R+0.587G+0.114B)",
                             callback=self.update_master_method)
                
                # Custom master formula (shown when Custom is selected)
                dpg.add_input_text(label="Custom Formula", tag="custom_master_formula", 
                                  default_value="0.299*r + 0.587*g + 0.114*b", show=False,
                                  hint="Use r, g, b variables (e.g., 0.33*r + 0.33*g + 0.34*b)")
            
            dpg.add_separator()
            
            # Art-Net settings
            with dpg.collapsing_header(label="Art-Net Settings", default_open=True):
                dpg.add_input_text(label="Target IP", default_value="255.255.255.255", tag="artnet_ip", width=150,
                                 callback=self.update_artnet_settings)
                dpg.add_input_int(label="Port", default_value=6454, tag="artnet_port", width=100,
                                callback=self.update_artnet_settings)
                dpg.add_input_int(label="Universe", default_value=0, tag="artnet_universe", width=100)
                dpg.add_slider_int(label="Brightness", default_value=255, min_value=0, max_value=255, tag="brightness")
                
                dpg.add_checkbox(label="Debug Mode", tag="artnet_debug", callback=self.toggle_artnet_debug)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Connect", callback=self.connect_artnet)
                    dpg.add_button(label="Disconnect", callback=self.disconnect_artnet)
                    dpg.add_button(label="Send Poll", callback=self.send_artnet_poll)
                
                # Art-Net status info
                dpg.add_separator()
                dpg.add_text("Art-Net Status:")
                dpg.add_text("Packets Sent: 0", tag="artnet_packets")
                dpg.add_text("Sequence: 0", tag="artnet_sequence")
                dpg.add_text("Last Error: None", tag="artnet_error")
            
            dpg.add_separator()
            
            # Color settings
            with dpg.collapsing_header(label="Color Settings"):
                dpg.add_slider_float(label="Gamma", default_value=1.0, min_value=0.1, max_value=3.0, tag="gamma")
                dpg.add_slider_float(label="Saturation", default_value=1.0, min_value=0.0, max_value=2.0, tag="saturation")
                dpg.add_checkbox(label="RGB to BGR", tag="rgb_to_bgr")
            
            dpg.add_separator()
            
            # Debug tools
            with dpg.collapsing_header(label="Debug Tools"):
                dpg.add_button(label="Test Pixel Selection", callback=self.test_pixel_selection, width=150)
                dpg.add_button(label="Toggle Pixel Debug", callback=self.toggle_pixel_debug, width=150)
                dpg.add_text("Use these tools to troubleshoot pixel mapping issues")
    
    def create_pixel_mapping_window(self):
        """Create pixel mapping window - updated for master channel"""
        with dpg.window(label="Pixel Mapping", tag="pixel_mapping_window", 
                       width=480, height=550, show=False, pos=[840, 20]):
            
            # Mapping mode toggle
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Mapping Mode", tag="mapping_mode", 
                               callback=self.toggle_mapping_mode)
                dpg.add_text("Click pixels to select", tag="mapping_hint")
            
            dpg.add_separator()
            
            # Selected pixel info
            with dpg.collapsing_header(label="Selected Pixel", default_open=True):
                dpg.add_text("Pixel: None selected", tag="selected_pixel_info")
                dpg.add_text("RGB: N/A", tag="selected_pixel_rgb")
                dpg.add_text("Master: N/A", tag="selected_pixel_master")  # New: Master value display
                
                # Channel assignment for selected pixel - updated for master
                with dpg.group():
                    dpg.add_text("DMX Channel Assignment:")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("R:")
                        dpg.add_input_int(label="##r_channel", tag="r_channel", width=80, 
                                        callback=self.update_pixel_mapping)
                        dpg.add_button(label="Auto", callback=lambda: self.auto_assign_channel('r'), width=50)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("G:")
                        dpg.add_input_int(label="##g_channel", tag="g_channel", width=80,
                                        callback=self.update_pixel_mapping)
                        dpg.add_button(label="Auto", callback=lambda: self.auto_assign_channel('g'), width=50)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("B:")
                        dpg.add_input_int(label="##b_channel", tag="b_channel", width=80,
                                        callback=self.update_pixel_mapping)
                        dpg.add_button(label="Auto", callback=lambda: self.auto_assign_channel('b'), width=50)
                    
                    # New: Master channel assignment
                    with dpg.group(horizontal=True):
                        dpg.add_text("M:")
                        dpg.add_input_int(label="##m_channel", tag="m_channel", width=80,
                                        callback=self.update_pixel_mapping)
                        dpg.add_button(label="Auto", callback=lambda: self.auto_assign_channel('m'), width=50)
                        dpg.add_checkbox(label="Enable", tag="m_channel_enable", default_value=True)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Clear Mapping", callback=self.clear_selected_mapping)
                        dpg.add_button(label="Apply", callback=self.apply_pixel_mapping)
            
            dpg.add_separator()
            
            # Mapping table - updated for master channel
            with dpg.collapsing_header(label="All Mappings", default_open=True):
                with dpg.table(header_row=True, tag="mapping_table", height=200, scrollY=True):
                    dpg.add_table_column(label="Pixel", width_fixed=True, init_width_or_weight=60)
                    dpg.add_table_column(label="R Ch", width_fixed=True, init_width_or_weight=40)
                    dpg.add_table_column(label="G Ch", width_fixed=True, init_width_or_weight=40)
                    dpg.add_table_column(label="B Ch", width_fixed=True, init_width_or_weight=40)
                    dpg.add_table_column(label="M Ch", width_fixed=True, init_width_or_weight=40)  # New: Master column
                    dpg.add_table_column(label="Actions", width_fixed=True, init_width_or_weight=70)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Auto Map All", callback=self.auto_map_all_pixels, width=85)
                    dpg.add_button(label="Clear All", callback=self.clear_all_mappings, width=85)
                
                # File operations - JSON only
                dpg.add_separator()
                dpg.add_text("Save/Load Mappings:")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="💾 Export JSON", callback=self.export_mappings, width=120)
                    dpg.add_button(label="📁 Import JSON", callback=self.import_mappings, width=120)
                
                dpg.add_text("💡 JSON format preserves all settings and metadata", wrap=400)
            
            dpg.add_separator()
            
            # Quick mapping presets - updated for master channel
            with dpg.collapsing_header(label="Quick Presets"):
                dpg.add_text("Sequential mapping:")
                with dpg.group(horizontal=True):
                    dpg.add_text("Start channel:")
                    dpg.add_input_int(label="##start_channel", tag="start_channel", default_value=1, width=80)
                    dpg.add_checkbox(label="Include Master", tag="preset_include_master", default_value=True)
                    dpg.add_button(label="Apply Sequential", callback=self.apply_sequential_mapping)
                
                dpg.add_separator()
                dpg.add_text("Matrix mapping:")
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Snake pattern", tag="snake_pattern")
                    dpg.add_checkbox(label="Reverse rows", tag="reverse_rows")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Apply Matrix", callback=self.apply_matrix_mapping)
                    dpg.add_button(label="Preview Matrix", callback=self.preview_matrix_mapping)
    
    # New method: Update master setting
    def update_master_setting(self):
        """Update master channel setting from main window"""
        self.master_enabled = dpg.get_value("master_enabled_main")
        if dpg.does_item_exist("master_enabled"):
            dpg.set_value("master_enabled", self.master_enabled)
        status = "Enabled" if self.master_enabled else "Disabled"
        dpg.set_value("master_status", f"Master: {status}")
        
        # Update channel format display
        format_text = "RGBM" if self.master_enabled else "RGB"
        if dpg.does_item_exist("channel_format"):
            dpg.set_value("channel_format", f"Format: {format_text}")
    
    def update_master_enabled(self):
        """Update master enabled from settings window"""
        self.master_enabled = dpg.get_value("master_enabled")
        dpg.set_value("master_enabled_main", self.master_enabled)
        status = "Enabled" if self.master_enabled else "Disabled"
        dpg.set_value("master_status", f"Master: {status}")
        
        # Update radio button
        if dpg.does_item_exist("channel_format_radio"):
            dpg.set_value("channel_format_radio", 1 if self.master_enabled else 0)
        
        # Update channel format display
        format_text = "RGBM" if self.master_enabled else "RGB"
        if dpg.does_item_exist("channel_format"):
            dpg.set_value("channel_format", f"Format: {format_text}")
    
    def update_channel_format(self):
        """Update channel format from radio button"""
        format_value = dpg.get_value("channel_format_radio")
        self.master_enabled = (format_value == 1)  # 0=RGB, 1=RGBM
        dpg.set_value("master_enabled", self.master_enabled)
        dpg.set_value("master_enabled_main", self.master_enabled)
        status = "Enabled" if self.master_enabled else "Disabled"
        dpg.set_value("master_status", f"Master: {status}")
        
        # Update channel format display
        format_text = "RGBM" if self.master_enabled else "RGB"
        if dpg.does_item_exist("channel_format"):
            dpg.set_value("channel_format", f"Format: {format_text}")
    
    def update_master_method(self):
        """Update master calculation method"""
        method = dpg.get_value("master_method")
        show_custom = method == "Custom"
        if dpg.does_item_exist("custom_master_formula"):
            dpg.configure_item("custom_master_formula", show=show_custom)
    
    def calculate_master_value(self, r: int, g: int, b: int) -> int:
        """Calculate master value based on selected method"""
        try:
            method = dpg.get_value("master_method") if dpg.does_item_exist("master_method") else "Luminance (0.299R+0.587G+0.114B)"
            
            if "Luminance" in method:
                # Luminance formula (most accurate for human perception)
                master = int(0.299 * r + 0.587 * g + 0.114 * b)
            elif "Average" in method:
                # Simple average
                master = int((r + g + b) / 3)
            elif "Maximum" in method:
                # Maximum of RGB values
                master = max(r, g, b)
            elif "Custom" in method:
                # Custom formula
                formula = dpg.get_value("custom_master_formula") if dpg.does_item_exist("custom_master_formula") else "0.299*r + 0.587*g + 0.114*b"
                try:
                    # Replace variables and evaluate
                    formula = formula.replace('r', str(r)).replace('g', str(g)).replace('b', str(b))
                    master = int(eval(formula))
                except:
                    # Fallback to luminance if custom formula fails
                    master = int(0.299 * r + 0.587 * g + 0.114 * b)
            else:
                # Default to luminance
                master = int(0.299 * r + 0.587 * g + 0.114 * b)
            
            # Ensure value is in valid range
            return max(0, min(255, master))
        except Exception as e:
            print(f"Master calculation error: {e}")
            return int(0.299 * r + 0.587 * g + 0.114 * b)  # Fallback to luminance
    
    # Window toggle methods remain the same...
    def toggle_video_preview(self):
        """Toggle video preview window"""
        self.windows["video_preview"] = not self.windows["video_preview"]
        dpg.configure_item("video_preview_window", show=self.windows["video_preview"])
    
    def toggle_dmx_data(self):
        """Toggle DMX data window"""
        self.windows["dmx_data"] = not self.windows["dmx_data"]
        dpg.configure_item("dmx_data_window", show=self.windows["dmx_data"])
    
    def toggle_pixel_mapping(self):
        """Toggle pixel mapping window"""
        self.windows["pixel_mapping"] = not self.windows["pixel_mapping"]
        dpg.configure_item("pixel_mapping_window", show=self.windows["pixel_mapping"])
    
    def toggle_settings(self):
        """Toggle settings window"""
        self.windows["settings"] = not self.windows["settings"]
        dpg.configure_item("settings_window", show=self.windows["settings"])
    
    def toggle_audio_settings(self):
        """Toggle audio settings window"""
        self.windows["audio_settings"] = not self.windows["audio_settings"]
        dpg.configure_item("audio_settings_window", show=self.windows["audio_settings"])
    
    def show_about(self):
        """Show about dialog"""
        with dpg.window(label="About", modal=True, show=True, tag="about_window", 
                       width=350, height=320, pos=[300, 250]):
            dpg.add_text("pixcap - Video to DMX Art-Net Converter")
            dpg.add_text("Version 0.3.1 - Fixed Channels Edition")
            dpg.add_separator()
            dpg.add_text("Convert video pixels to DMX data")
            dpg.add_text("with RGB and Master brightness channels")
            dpg.add_text("and transmit via Art-Net protocol")
            dpg.add_text("Now with audio playback support!")
            dpg.add_separator()
            dpg.add_text("Features:")
            dpg.add_text("• RGB + Master channel output")
            dpg.add_text("• Fixed channels for dimmers/effects")
            dpg.add_text("• Multiple master calculation methods")
            dpg.add_text("• Custom pixel mapping")
            dpg.add_text("• Audio synchronization")
            dpg.add_text("• JSON settings export/import")
            dpg.add_separator()
            dpg.add_text("20250618")
            dpg.add_text("made by amane sasamoto")
            dpg.add_separator()
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("about_window"))
    
    # Art-Net methods remain mostly the same...
    def update_artnet_settings(self):
        """Update Art-Net settings"""
        ip = dpg.get_value("artnet_ip")
        port = dpg.get_value("artnet_port")
        self.artnet_controller.target_ip = ip
        self.artnet_controller.port = port
        self.artnet_controller._init_socket()
        status = "Connected" if self.artnet_controller.is_connected else "Disconnected"
        dpg.set_value("artnet_status", f"Art-Net: {status}")
    
    def toggle_artnet_debug(self):
        """Toggle Art-Net debug mode"""
        debug_enabled = dpg.get_value("artnet_debug")
        self.artnet_controller.enable_debug(debug_enabled)
        dpg.set_value("status_text", f"Art-Net debug mode: {'Enabled' if debug_enabled else 'Disabled'}")
    
    def send_artnet_poll(self):
        """Send Art-Net poll packet"""
        success = self.artnet_controller.send_artpoll()
        if success:
            dpg.set_value("status_text", "Art-Poll packet sent - check console for responses")
        else:
            dpg.set_value("status_text", "Failed to send Art-Poll packet")
    
    def update_artnet_stats(self):
        """Update Art-Net statistics display"""
        try:
            stats = self.artnet_controller.get_stats()
            dpg.set_value("artnet_packets", f"Packets Sent: {stats['packets_sent']}")
            dpg.set_value("artnet_sequence", f"Sequence: {stats['sequence']}")
            
            status = "Connected" if stats['connected'] else "Disconnected"
            dpg.set_value("artnet_status", f"Art-Net: {status}")
        except:
            pass
    
    def connect_artnet(self):
        """Connect to Art-Net"""
        self.artnet_controller._init_socket()
        if self.artnet_controller.is_connected:
            dpg.set_value("artnet_status", "Art-Net: Connected")
            dpg.set_value("status_text", f"Connected to Art-Net: {self.artnet_controller.target_ip}")
            dpg.set_value("artnet_error", "Last Error: None")
        else:
            dpg.set_value("artnet_status", "Art-Net: Failed")
            dpg.set_value("status_text", "Failed to connect to Art-Net")
            dpg.set_value("artnet_error", "Last Error: Socket initialization failed")
    
    def disconnect_artnet(self):
        """Disconnect from Art-Net"""
        if self.artnet_controller.socket:
            self.artnet_controller.socket.close()
        self.artnet_controller.is_connected = False
        dpg.set_value("artnet_status", "Art-Net: Disconnected")
        dpg.set_value("status_text", "Art-Net disconnected")
        dpg.set_value("artnet_error", "Last Error: None")
    
    # Audio methods remain the same...
    def change_audio_device(self, sender, app_data):
        """Change audio output device"""
        selected_device = app_data
        self.audio_controller.selected_device = selected_device
        dpg.set_value("status_text", f"Audio device set to: {selected_device}")
    
    def toggle_audio_playback(self):
        """Toggle audio playback"""
        self.audio_enabled = dpg.get_value("enable_audio")
        dpg.set_value("audio_enabled_main", self.audio_enabled)
    
    def set_audio_volume(self, sender, volume):
        """Set audio volume"""
        self.audio_controller.set_volume(volume)
        dpg.set_value("volume_slider", volume)
        dpg.set_value("volume_display", f"{int(volume * 100)}%")
    
    def update_volume(self, sender, volume):
        """Update volume from main window"""
        self.audio_controller.set_volume(volume)
        if dpg.does_item_exist("audio_volume"):
            dpg.set_value("audio_volume", volume)
        dpg.set_value("volume_display", f"{int(volume * 100)}%")
    
    def test_audio(self):
        """Test audio playback"""
        if self.audio_controller.sound:
            self.audio_controller.play()
            dpg.set_value("status_text", "Testing audio playback...")
        else:
            dpg.set_value("status_text", "No audio loaded to test")
    
    def refresh_audio_devices(self):
        """Refresh audio device list"""
        self.audio_controller._get_audio_devices()
        device_names = self.audio_controller.available_devices
        dpg.configure_item("audio_device_combo", items=device_names)
        dpg.set_value("status_text", f"Found {len(device_names)} audio devices")
    
    def update_loop_setting(self):
        """Update loop setting from main window"""
        self.loop_playback = dpg.get_value("loop_playback_main")
        if dpg.does_item_exist("loop_playback"):
            dpg.set_value("loop_playbook", self.loop_playback)
    
    def update_audio_setting(self):
        """Update audio setting from main window"""
        self.audio_enabled = dpg.get_value("audio_enabled_main")
        if dpg.does_item_exist("enable_audio"):
            dpg.set_value("enable_audio", self.audio_enabled)
    
    def file_selected(self, sender, app_data):
        """File selection callback"""
        file_path = app_data['file_path_name']
        if self.video_processor.load_video(file_path):
            self.current_file = file_path
            filename = os.path.basename(file_path)
            dpg.set_value("file_status", f"File: {filename}")
            dpg.set_value("frame_info", f"Frame: 0/{self.video_processor.frame_count}")
            dpg.set_value("status_text", "Video loaded successfully")
            
            # Update video info
            width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dpg.set_value("video_resolution", f"Resolution: {width}x{height}")
            dpg.set_value("video_fps", f"FPS: {self.video_processor.fps:.2f}")
            
            # Format duration
            duration = self.video_processor.duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            dpg.set_value("duration_info", f"Duration: {minutes:02d}:{seconds:02d}")
            if dpg.does_item_exist("video_duration"):
                dpg.set_value("video_duration", f"Duration: {minutes:02d}:{seconds:02d}")
            
            # Update seek slider maximum
            dpg.configure_item("seek_slider", max_value=duration)
            
            # Load audio
            if self.audio_enabled:
                if self.audio_controller.load_audio(file_path):
                    dpg.set_value("audio_status", "Audio: Loaded")
                    dpg.set_value("audio_info_status", "Status: Loaded")
                else:
                    dpg.set_value("audio_status", "Audio: Failed to load")
                    dpg.set_value("audio_info_status", "Status: Failed to load")
            else:
                dpg.set_value("audio_status", "Audio: Disabled")
            
            # Show first frame
            frame = self.video_processor.get_frame(0)
            if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                self.update_preview(frame)
        else:
            dpg.set_value("status_text", "Failed to load video")
    
    def seek_video(self, sender, position):
        """Seek to position in video"""
        if not self.video_processor.cap:
            return
        
        duration = self.video_processor.duration
        time_seconds = position * duration
        
        # Seek video
        self.video_processor.seek_to_time(time_seconds)
        
        # Seek audio if playing
        if self.audio_enabled and self.audio_controller.is_playing:
            self.audio_controller.stop()
            self.audio_controller.play(time_seconds)
        
        # Update display
        frame = self.video_processor.get_frame()
        if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
            self.update_preview(frame)
        
        # Update time display
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        dpg.set_value("time_display", f"{minutes:02d}:{seconds:02d}")
    
    def restart_video(self):
        """Restart video from beginning"""
        if not self.current_file:
            dpg.set_value("status_text", "No video file loaded")
            return
        
        # Stop current playback
        self.stop_video()
        
        # Reset to beginning
        self.video_processor.current_frame = 0
        self.video_processor.seek_to_time(0)
        
        # Reset UI
        dpg.set_value("seek_slider", 0.0)
        dpg.set_value("time_display", "00:00")
        dpg.set_value("progress", 0.0)
        
        # Show first frame
        frame = self.video_processor.get_frame(0)
        if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
            self.update_preview(frame)
        
        dpg.set_value("status_text", "Video restarted")
    
    def format_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def update_preview(self, frame: np.ndarray):
        """Update preview image with zoom and pan support - Fixed coordinate system"""
        try:
            # Validate frame data
            if frame is None or not hasattr(frame, 'shape') or frame.size == 0:
                print("Invalid frame data provided to update_preview")
                return
            
            orig_height, orig_width = frame.shape[:2]
            
            # Create 320x240 canvas
            canvas = np.zeros((240, 320, 3), dtype=np.uint8)
            
            # Calculate the region of the original video that should be visible
            # considering zoom and pan
            
            # How many original video pixels fit in the 320x240 preview?
            visible_video_width = 320 / self.zoom_level
            visible_video_height = 240 / self.zoom_level
            
            # What's the top-left corner of the visible area in video coordinates?
            video_start_x = self.pan_x / self.zoom_level
            video_start_y = self.pan_y / self.zoom_level
            
            # Calculate the actual region to sample from the video
            video_x1 = max(0, int(video_start_x))
            video_y1 = max(0, int(video_start_y))
            video_x2 = min(orig_width, int(video_start_x + visible_video_width) + 1)
            video_y2 = min(orig_height, int(video_start_y + visible_video_height) + 1)
            
            # Extract the visible region from the original video
            if video_x2 > video_x1 and video_y2 > video_y1:
                video_region = frame[video_y1:video_y2, video_x1:video_x2]
                
                # Calculate where this region should be placed on the canvas
                canvas_x1 = max(0, int((video_x1 - video_start_x) * self.zoom_level))
                canvas_y1 = max(0, int((video_y1 - video_start_y) * self.zoom_level))
                
                # Resize the video region to match the zoom level
                target_width = int((video_x2 - video_x1) * self.zoom_level)
                target_height = int((video_y2 - video_y1) * self.zoom_level)
                
                if target_width > 0 and target_height > 0:
                    # Use INTER_NEAREST for pixel-perfect scaling
                    zoomed_region = cv2.resize(video_region, (target_width, target_height), 
                                             interpolation=cv2.INTER_NEAREST)
                    
                    # Calculate actual placement on canvas
                    canvas_x2 = min(320, canvas_x1 + target_width)
                    canvas_y2 = min(240, canvas_y1 + target_height)
                    
                    # Adjust zoomed region if it exceeds canvas
                    if canvas_x2 > canvas_x1 and canvas_y2 > canvas_y1:
                        region_width = canvas_x2 - canvas_x1
                        region_height = canvas_y2 - canvas_y1
                        
                        if region_width != target_width or region_height != target_height:
                            zoomed_region = zoomed_region[:region_height, :region_width]
                        
                        canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = zoomed_region
            
            # Add grid overlay if enabled
            if dpg.get_value("show_grid") if dpg.does_item_exist("show_grid") else False:
                self._draw_grid_overlay(canvas, orig_width, orig_height)
            
            # Highlight selected pixel
            if self.mapping_mode and self.selected_pixel is not None:
                self._draw_pixel_highlight(canvas, orig_width, orig_height)
            
            # Show mapped pixels
            if self.mapping_mode and len(self.pixel_mappings) > 0:
                self._draw_mapped_pixels(canvas, orig_width, orig_height)
            
            # Update display info
            self._update_display_info()
            
            # Convert to RGBA and update texture
            rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
            normalized = rgba.astype(np.float32) / 255.0
            dpg.set_value("video_texture", normalized.flatten())
            
        except Exception as e:
            print(f"Preview update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_grid_overlay(self, canvas, orig_width, orig_height):
        """Draw grid overlay on canvas"""
        try:
            # Get grid appearance settings
            if dpg.does_item_exist("grid_color"):
                grid_color_float = dpg.get_value("grid_color")
                grid_color = (int(grid_color_float[2] * 255), int(grid_color_float[1] * 255), int(grid_color_float[0] * 255))  # BGR
            else:
                grid_color = (128, 128, 128)  # Default gray
            
            line_width = dpg.get_value("grid_line_width") if dpg.does_item_exist("grid_line_width") else 1
            
            # Draw lines for each video pixel boundary
            for video_x in range(orig_width + 1):
                # Convert video coordinate to preview coordinate
                preview_x = int(video_x * self.zoom_level - self.pan_x)
                if 0 <= preview_x < 320:
                    cv2.line(canvas, (preview_x, 0), (preview_x, 240), grid_color, line_width)
            
            for video_y in range(orig_height + 1):
                # Convert video coordinate to preview coordinate  
                preview_y = int(video_y * self.zoom_level - self.pan_y)
                if 0 <= preview_y < 240:
                    cv2.line(canvas, (0, preview_y), (320, preview_y), grid_color, line_width)
                    
        except Exception as e:
            print(f"Grid overlay error: {e}")
    
    def _draw_pixel_highlight(self, canvas, orig_width, orig_height):
        """Draw highlight for selected pixel"""
        try:
            sel_x, sel_y = self.selected_pixel
            
            # Convert video pixel to preview coordinates
            preview_x1 = int(sel_x * self.zoom_level - self.pan_x)
            preview_y1 = int(sel_y * self.zoom_level - self.pan_y)
            preview_x2 = int((sel_x + 1) * self.zoom_level - self.pan_x)
            preview_y2 = int((sel_y + 1) * self.zoom_level - self.pan_y)
            
            # Only draw if visible in canvas
            if (preview_x1 < 320 and preview_y1 < 240 and preview_x2 > 0 and preview_y2 > 0):
                # Clamp to canvas boundaries
                preview_x1 = max(0, preview_x1)
                preview_y1 = max(0, preview_y1)
                preview_x2 = min(320, preview_x2)
                preview_y2 = min(240, preview_y2)
                
                # Draw yellow border for selected pixel
                cv2.rectangle(canvas, (preview_x1, preview_y1), (preview_x2, preview_y2), 
                            (0, 255, 255), 2)  # Yellow border
                            
        except Exception as e:
            print(f"Pixel highlight error: {e}")
    
    def _draw_mapped_pixels(self, canvas, orig_width, orig_height):
        """Draw indicators for mapped pixels"""
        try:
            for (px, py), mapping in self.pixel_mappings.items():
                # Skip the currently selected pixel (already highlighted in yellow)
                if self.selected_pixel is None or (px != self.selected_pixel[0] or py != self.selected_pixel[1]):
                    # Convert video pixel to preview coordinates
                    preview_x1 = int(px * self.zoom_level - self.pan_x)
                    preview_y1 = int(py * self.zoom_level - self.pan_y)
                    preview_x2 = int((px + 1) * self.zoom_level - self.pan_x)
                    preview_y2 = int((py + 1) * self.zoom_level - self.pan_y)
                    
                    # Only draw if visible in canvas
                    if (preview_x1 < 320 and preview_y1 < 240 and preview_x2 > 0 and preview_y2 > 0):
                        # Clamp to canvas boundaries
                        preview_x1 = max(0, preview_x1)
                        preview_y1 = max(0, preview_y1)
                        preview_x2 = min(320, preview_x2)
                        preview_y2 = min(240, preview_y2)
                        
                        # Draw green border for mapped pixels
                        cv2.rectangle(canvas, (preview_x1, preview_y1), (preview_x2, preview_y2), 
                                    (0, 255, 0), 1)  # Green border
                        
                        # Show pixel info for larger pixels
                        pixel_width = preview_x2 - preview_x1
                        if pixel_width > 25:
                            text_x = preview_x1 + int(pixel_width * 0.1)
                            text_y = preview_y1 + int((preview_y2 - preview_y1) * 0.7)
                            
                            if 0 <= text_x < 300 and 10 <= text_y < 240:
                                # Show mapping channel info
                                r_ch = mapping.get('r', 0)
                                text = f"R{r_ch}" if r_ch > 0 else "M"
                                
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = min(0.4, pixel_width / 60)
                                text_color = (255, 255, 255)  # White text
                                bg_color = (0, 128, 0)  # Dark green background
                                
                                # Get text size for background
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 1)
                                
                                # Draw background rectangle
                                cv2.rectangle(canvas, 
                                            (text_x - 1, text_y - text_height - 1), 
                                            (text_x + text_width + 1, text_y + 1), 
                                            bg_color, -1)
                                
                                # Draw text
                                cv2.putText(canvas, text, (text_x, text_y), 
                                          font, font_scale, text_color, 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Mapped pixels drawing error: {e}")
    
    def _update_display_info(self):
        """Update display information"""
        try:
            if dpg.does_item_exist("scale_factor"):
                dpg.set_value("scale_factor", f"Scale Factor: {self.zoom_level:.1f}x")
            
            if dpg.does_item_exist("current_frame_info"):
                dpg.set_value("current_frame_info", f"Current Frame: {self.video_processor.current_frame}")
            
            if dpg.does_item_exist("view_area"):
                dpg.set_value("view_area", f"View Area: {self.pan_x:.0f},{self.pan_y:.0f}")
        except Exception as e:
            print(f"Display info update error: {e}")
    
    def playback_worker(self):
        """Playback worker thread with audio sync"""
        if not self.video_processor.cap:
            return
        
        fps = dpg.get_value("fps_override")
        frame_delay = 1.0 / fps
        
        # Start audio if enabled
        if self.audio_enabled and self.audio_controller.sound:
            self.audio_controller.play()
        
        start_time = time.time()
        last_data_rate_update = start_time
        
        while not self.stop_playback:
            frame_start_time = time.time()
            
            # Check if we've reached the end
            if self.video_processor.current_frame >= self.video_processor.frame_count:
                if self.loop_playback:
                    # Restart from beginning
                    self.video_processor.current_frame = 0
                    self.video_processor.seek_to_time(0)
                    if self.audio_enabled and self.audio_controller.sound:
                        self.audio_controller.stop()
                        self.audio_controller.play()
                    start_time = time.time()
                else:
                    break
            
            # Get frame
            frame = self.video_processor.get_frame()
            if frame is None:
                break
            
            try:
                # Extract pixel data with master channel support
                grid_w = dpg.get_value("grid_width") if dpg.does_item_exist("grid_width") else 16
                grid_h = dpg.get_value("grid_height") if dpg.does_item_exist("grid_height") else 16
                pixel_data = self.video_processor.extract_pixel_data(frame, (grid_w, grid_h), self.master_enabled)
                
                # Convert to DMX
                dmx_data = self.pixels_to_dmx(pixel_data)
                self.current_dmx_data = dmx_data
                
                # Send Art-Net
                universe = dpg.get_value("artnet_universe") if dpg.does_item_exist("artnet_universe") else 0
                target_ip = dpg.get_value("artnet_ip") if dpg.does_item_exist("artnet_ip") else "255.255.255.255"
                self.artnet_controller.target_ip = target_ip
                
                success = self.artnet_controller.send_dmx(dmx_data, universe)
                if not success:
                    print(f"Failed to send Art-Net data (frame {self.video_processor.current_frame})")
                
                # Update UI
                current_time = self.video_processor.get_current_time()
                progress = current_time / self.video_processor.duration if self.video_processor.duration > 0 else 0
                
                dpg.set_value("progress", progress)
                dpg.set_value("frame_info", f"Frame: {self.video_processor.current_frame}/{self.video_processor.frame_count}")
                channels_per_pixel = 4 if self.master_enabled else 3
                pixel_count = len(dmx_data) // channels_per_pixel
                dpg.set_value("dmx_info", f"DMX Channels: {len(dmx_data)} ({pixel_count} pixels)")
                dpg.set_value("status_text", "Playing")
                dpg.set_value("seek_slider", current_time)
                dpg.set_value("time_display", self.format_time(current_time))
                
                # Update preview and DMX table
                if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                    self.update_preview(frame)
                self.update_dmx_table(dmx_data)
                
                # Update Art-Net statistics
                self.update_artnet_stats()
                
                # Calculate data rate (every second)
                if time.time() - last_data_rate_update >= 1.0:
                    if dpg.does_item_exist("data_rate"):
                        dpg.set_value("data_rate", f"Data Rate: {fps:.1f} Hz")
                    last_data_rate_update = time.time()
                
            except Exception as e:
                print(f"Playback error: {e}")
            
            # Frame rate control
            elapsed = time.time() - frame_start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self.video_processor.current_frame += 1
        
        # Playback finished
        if self.audio_enabled:
            self.audio_controller.stop()
        
        dpg.set_value("status_text", "Playback finished" if not self.stop_playback else "Playback stopped")
        dpg.configure_item("play_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)
    
    def play_video(self):
        """Start video playback"""
        if not self.current_file:
            dpg.set_value("status_text", "Please select a video file")
            return
        
        self.stop_playback = False
        dpg.configure_item("play_btn", enabled=False)
        dpg.configure_item("stop_btn", enabled=True)
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def stop_video(self):
        """Stop video playback"""
        self.stop_playback = True
        if self.audio_enabled:
            self.audio_controller.stop()
        dpg.configure_item("play_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)
    
    def apply_fixed_channels(self, dmx_data: List[int]) -> List[int]:
        """Apply fixed channel values to DMX data"""
        try:
            if not self.fixed_channels:
                return dmx_data
            
            # Ensure DMX data is at least 512 channels
            result = dmx_data.copy()
            if len(result) < 512:
                result.extend([0] * (512 - len(result)))
            elif len(result) > 512:
                result = result[:512]
            
            # Apply enabled fixed channels
            applied_count = 0
            for channel, info in self.fixed_channels.items():
                if info['enabled'] and 1 <= channel <= 512:
                    # Convert to 0-based index
                    index = channel - 1
                    result[index] = info['value']
                    applied_count += 1
            
            if applied_count > 0:
                print(f"Applied {applied_count} fixed channels to DMX data")
                
            return result
            
        except Exception as e:
            print(f"Apply fixed channels error: {e}")
            return dmx_data
    
    def pixels_to_dmx(self, pixel_data: List[List[int]]) -> List[int]:
        """Convert pixel data to DMX data using custom mappings - Updated with fixed channels"""
        try:
            # Get base DMX data from pixel mappings
            if len(self.pixel_mappings) > 0:
                base_dmx = self.pixels_to_dmx_custom_mapping(pixel_data)
            else:
                base_dmx = self.pixels_to_dmx_sequential(pixel_data)
            
            # Apply fixed channels
            final_dmx = self.apply_fixed_channels(base_dmx)
            
            return final_dmx
            
        except Exception as e:
            print(f"Pixels to DMX conversion error: {e}")
            return []
    
    def pixels_to_dmx_sequential(self, pixel_data: List[List[int]]) -> List[int]:
        """Convert pixel data to DMX data sequentially - updated for master channel"""
        try:
            dmx_data = []
            brightness = dpg.get_value("brightness") if dpg.does_item_exist("brightness") else 255
            gamma = dpg.get_value("gamma") if dpg.does_item_exist("gamma") else 1.0
            
            for pixel in pixel_data:
                if self.master_enabled and len(pixel) >= 4:
                    r, g, b, m = pixel[0], pixel[1], pixel[2], pixel[3]
                elif len(pixel) >= 3:
                    r, g, b = pixel[0], pixel[1], pixel[2]
                    m = self.calculate_master_value(r, g, b) if self.master_enabled else 0
                else:
                    continue  # Skip invalid pixel data
                
                # Apply gamma correction
                r = int(255 * pow(r / 255.0, gamma))
                g = int(255 * pow(g / 255.0, gamma))
                b = int(255 * pow(b / 255.0, gamma))
                if self.master_enabled:
                    m = int(255 * pow(m / 255.0, gamma))
                
                # Apply brightness
                r = int(r * brightness / 255)
                g = int(g * brightness / 255)
                b = int(b * brightness / 255)
                if self.master_enabled:
                    m = int(m * brightness / 255)
                
                # Add to DMX data
                if self.master_enabled:
                    dmx_data.extend([r, g, b, m])
                else:
                    dmx_data.extend([r, g, b])
            
            return dmx_data
        except Exception as e:
            print(f"Sequential DMX conversion error: {e}")
            return []
    
    def pixels_to_dmx_custom_mapping(self, pixel_data: List[List[int]]) -> List[int]:
        """Convert pixel data to DMX using custom pixel mappings - updated for master channel"""
        try:
            # Check if video processor and frame data are available
            if not self.video_processor.cap:
                return []
            
            if self.video_processor.frame_data is None:
                return []
            
            # Additional check for empty array
            if not hasattr(self.video_processor.frame_data, 'shape') or self.video_processor.frame_data.size == 0:
                return []
            
            # Get current frame for direct pixel access
            frame = self.video_processor.frame_data
            orig_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Enable debug only when requested
            debug_enabled = (hasattr(self, '_debug_pixel_extraction') and 
                           self._debug_pixel_extraction and 
                           hasattr(self, '_debug_custom_mapping_count'))
            
            if not hasattr(self, '_debug_custom_mapping_count'):
                self._debug_custom_mapping_count = 0
            
            self._debug_custom_mapping_count += 1
            
            if debug_enabled and self._debug_custom_mapping_count <= 3:
                print(f"\n=== Custom DMX Mapping #{self._debug_custom_mapping_count} ===")
                print(f"Frame: {orig_width}x{orig_height}")
                print(f"Mappings: {len(self.pixel_mappings)}")
                print(f"Master enabled: {self.master_enabled}")
            
            # Find the maximum channel number to size our DMX array
            max_channel = 0
            for mapping in self.pixel_mappings.values():
                for ch in mapping.values():
                    if ch > max_channel:
                        max_channel = ch
            
            if max_channel == 0:
                if debug_enabled:
                    print("No custom mappings found, using sequential")
                return self.pixels_to_dmx_sequential(pixel_data)
            
            # Initialize DMX array with zeros (will be padded to 512 in send_dmx)
            dmx_data = [0] * min(max_channel, 512)  # Limit to 512 channels
            
            brightness = dpg.get_value("brightness") if dpg.does_item_exist("brightness") else 255
            gamma = dpg.get_value("gamma") if dpg.does_item_exist("gamma") else 1.0
            
            if debug_enabled and self._debug_custom_mapping_count <= 3:
                print(f"DMX array size: {len(dmx_data)}")
                print(f"Brightness: {brightness}, Gamma: {gamma}")
            
            # Process each mapped pixel using direct frame access
            mapped_count = 0
            for (px, py), mapping in self.pixel_mappings.items():
                if 0 <= px < orig_width and 0 <= py < orig_height:
                    # Get pixel value directly from frame at exact coordinates
                    b, g, r = frame[py, px]  # Note: OpenCV uses BGR and y,x indexing
                    
                    # Calculate master value if enabled
                    if self.master_enabled:
                        master = self.calculate_master_value(r, g, b)
                    
                    # Apply gamma correction
                    r = int(255 * pow(r / 255.0, gamma))
                    g = int(255 * pow(g / 255.0, gamma))
                    b = int(255 * pow(b / 255.0, gamma))
                    if self.master_enabled:
                        master = int(255 * pow(master / 255.0, gamma))
                    
                    # Apply brightness
                    r = int(r * brightness / 255)
                    g = int(g * brightness / 255)
                    b = int(b * brightness / 255)
                    if self.master_enabled:
                        master = int(master * brightness / 255)
                    
                    # Set DMX channels (convert from 1-based to 0-based indexing)
                    if mapping['r'] > 0 and mapping['r'] <= len(dmx_data):
                        dmx_data[mapping['r'] - 1] = r
                    if mapping['g'] > 0 and mapping['g'] <= len(dmx_data):
                        dmx_data[mapping['g'] - 1] = g
                    if mapping['b'] > 0 and mapping['b'] <= len(dmx_data):
                        dmx_data[mapping['b'] - 1] = b
                    # Set master channel if enabled and mapped
                    if self.master_enabled and mapping.get('m', 0) > 0 and mapping['m'] <= len(dmx_data):
                        dmx_data[mapping['m'] - 1] = master
                    
                    mapped_count += 1
                    
                    # Debug first few mappings only
                    if debug_enabled and self._debug_custom_mapping_count <= 3 and mapped_count <= 3:
                        if self.master_enabled:
                            print(f"Pixel ({px},{py}): frame[{py},{px}] = BGR({b},{g},{r}) -> RGB({r},{g},{b}) Master({master})")
                            print(f"  -> DMX channels R:{mapping.get('r',0)} G:{mapping.get('g',0)} B:{mapping.get('b',0)} M:{mapping.get('m',0)}")
                        else:
                            print(f"Pixel ({px},{py}): frame[{py},{px}] = BGR({b},{g},{r}) -> RGB({r},{g},{b})")
                            print(f"  -> DMX channels R:{mapping.get('r',0)} G:{mapping.get('g',0)} B:{mapping.get('b',0)}")
            
            if debug_enabled and self._debug_custom_mapping_count <= 3:
                print(f"Mapped {mapped_count} pixels")
                print("=== End Custom Mapping ===\n")
            
            return dmx_data
        except Exception as e:
            print(f"Custom DMX mapping error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_artnet(self):
        """Test Art-Net connection with enhanced RGBM pattern"""
        try:
            # Enable debug mode for test
            original_debug = self.artnet_controller.debug_mode
            self.artnet_controller.enable_debug(True)
            
            # Create comprehensive test pattern using RGBM format
            test_data = []
            
            if self.master_enabled:
                # Pattern 1: Rainbow pixels with master (first 80 channels = 20 RGBM pixels)
                rainbow_pixels = 20
                for i in range(rainbow_pixels):
                    hue = (i * 360 / rainbow_pixels) % 360
                    
                    # Convert HSV to RGB
                    import colorsys
                    r, g, b = colorsys.hsv_to_rgb(hue/360.0, 1.0, 1.0)
                    
                    # Convert to 0-255 range
                    r = int(r * 255)
                    g = int(g * 255)
                    b = int(b * 255)
                    
                    # Calculate master value
                    master = self.calculate_master_value(r, g, b)
                    
                    test_data.extend([r, g, b, master])
                
                # Pattern 2: Gradient fade with master (next 80 channels = 20 RGBM pixels)
                fade_pixels = 20
                for i in range(fade_pixels):
                    intensity = int(255 * (i / (fade_pixels - 1)))
                    r, g, b = intensity, intensity // 2, 255 - intensity
                    master = self.calculate_master_value(r, g, b)
                    test_data.extend([r, g, b, master])
                
                # Pattern 3: White levels with master (next 40 channels = 10 RGBM pixels)
                white_pixels = 10
                for i in range(white_pixels):
                    level = int(255 * ((i + 1) / white_pixels))
                    r, g, b = level, level, level
                    master = self.calculate_master_value(r, g, b)
                    test_data.extend([r, g, b, master])
                
                # Pattern 4: Primary colors with master (next 40 channels = 10 RGBM pixels)
                primary_pixels = 10
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                for i in range(primary_pixels):
                    r, g, b = colors[i % len(colors)]
                    master = self.calculate_master_value(r, g, b)
                    test_data.extend([r, g, b, master])
                
                # At this point we have 240 channels used (60 RGBM pixels)
                pattern_desc = "RGBM (RGB + Master)"
                pixels_desc = f"{len(test_data)//4} RGBM pixels"
                
            else:
                # RGB-only patterns
                # Pattern 1: Rainbow pixels (first 60 channels = 20 RGB pixels)
                rainbow_pixels = 20
                for i in range(rainbow_pixels):
                    hue = (i * 360 / rainbow_pixels) % 360
                    
                    # Convert HSV to RGB
                    import colorsys
                    r, g, b = colorsys.hsv_to_rgb(hue/360.0, 1.0, 1.0)
                    
                    # Convert to 0-255 range
                    r = int(r * 255)
                    g = int(g * 255)
                    b = int(b * 255)
                    
                    test_data.extend([r, g, b])
                
                # Pattern 2: Gradient fade (next 60 channels = 20 RGB pixels)
                fade_pixels = 20
                for i in range(fade_pixels):
                    intensity = int(255 * (i / (fade_pixels - 1)))
                    test_data.extend([intensity, intensity // 2, 255 - intensity])
                
                # Pattern 3: White levels (next 30 channels = 10 RGB pixels)
                white_pixels = 10
                for i in range(white_pixels):
                    level = int(255 * ((i + 1) / white_pixels))
                    test_data.extend([level, level, level])
                
                # Pattern 4: Primary colors cycling (next 30 channels = 10 RGB pixels)
                primary_pixels = 10
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                for i in range(primary_pixels):
                    color = colors[i % len(colors)]
                    test_data.extend(list(color))
                
                # At this point we have 180 channels used (60 RGB pixels)
                pattern_desc = "RGB"
                pixels_desc = f"{len(test_data)//3} RGB pixels"
            
            print(f"\n=== Art-Net 512-Channel Test Pattern ({pattern_desc}) ===")
            print(f"Pattern designed: {len(test_data)} channels")
            print(f"Will be padded to: 512 channels (DMX standard)")
            print(f"Format: {pattern_desc}")
            print(f"Pixels: {pixels_desc}")
            print(f"Patterns included:")
            if self.master_enabled:
                print(f"  - Rainbow: channels 1-80 (20 RGBM pixels)")
                print(f"  - Gradient: channels 81-160 (20 RGBM pixels)")
                print(f"  - White levels: channels 161-200 (10 RGBM pixels)")
                print(f"  - Primary colors: channels 201-240 (10 RGBM pixels)")
                print(f"  - Padding: channels 241-512 (zeros)")
            else:
                print(f"  - Rainbow: channels 1-60 (20 RGB pixels)")
                print(f"  - Gradient: channels 61-120 (20 RGB pixels)")
                print(f"  - White levels: channels 121-150 (10 RGB pixels)")
                print(f"  - Primary colors: channels 151-180 (10 RGB pixels)")
                print(f"  - Padding: channels 181-512 (zeros)")
            
            # Get settings
            universe = dpg.get_value("artnet_universe") if dpg.does_item_exist("artnet_universe") else 0
            target_ip = dpg.get_value("artnet_ip") if dpg.does_item_exist("artnet_ip") else "255.255.255.255"
            
            # Update controller settings
            self.artnet_controller.target_ip = target_ip
            
            print(f"Target: {target_ip}:6454")
            print(f"Universe: {universe}")
            
            # Send test pattern with fixed channels
            success = self.artnet_controller.send_dmx(test_data, universe)
            
            # Apply fixed channels to test data for display
            final_test_data = self.apply_fixed_channels(test_data)
            
            if success:
                channels_per_pixel = 4 if self.master_enabled else 3
                pixel_count = len(test_data) // channels_per_pixel
                fixed_count = len([ch for ch in self.fixed_channels.values() if ch['enabled']])
                
                dpg.set_value("status_text", f"✓ 512-channel {pattern_desc} test pattern sent (pixels: {pixel_count}, fixed: {fixed_count})")
                dpg.set_value("dmx_info", f"DMX Channels: 512 (active: {len(test_data)}, {pixel_count} pixels, {fixed_count} fixed)")
                dpg.set_value("artnet_status", "Art-Net: Connected")
                dpg.set_value("artnet_error", "Last Error: None")
                
                # Show first few values in status
                if self.master_enabled:
                    preview = f"RGBM[1]: ({test_data[0]},{test_data[1]},{test_data[2]},M{test_data[3]})"
                else:
                    preview = f"RGB[1]: ({test_data[0]},{test_data[1]},{test_data[2]})"
                    
                if fixed_count > 0:
                    fixed_preview = []
                    for ch, info in sorted(self.fixed_channels.items()):
                        if info['enabled']:
                            fixed_preview.append(f"Ch{ch}:{info['value']}")
                    preview += f" | Fixed: {', '.join(fixed_preview[:3])}"
                    if len(fixed_preview) > 3:
                        preview += "..."
                
                print(f"Success! {preview}")
                print(f"Packet size: 530 bytes (18 header + 512 data)")
                
            else:
                dpg.set_value("status_text", "✗ Failed to send test pattern")
                dpg.set_value("artnet_status", "Art-Net: Failed")
                dpg.set_value("artnet_error", "Last Error: Test pattern send failed")
                print("Failed to send test pattern")
            
            # Update DMX table with final test data (includes fixed channels)
            self.update_dmx_table(final_test_data)
            
            # Update statistics
            self.update_artnet_stats()
            
            # Restore original debug mode
            self.artnet_controller.enable_debug(original_debug)
            
            print("=== End Test ===\n")
            
        except Exception as e:
            print(f"Art-Net test error: {e}")
            dpg.set_value("status_text", f"Art-Net test failed: {e}")
            dpg.set_value("artnet_error", f"Last Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_dmx_table(self, dmx_data: List[int]):
        """Update DMX data table with RGBM support"""
        if not self.windows["dmx_data"]:
            return
        
        try:
            # Pad to 512 channels for display (matching what we send)
            display_data = dmx_data.copy()
            if len(display_data) < 512:
                display_data.extend([0] * (512 - len(display_data)))
            elif len(display_data) > 512:
                display_data = display_data[:512]
            
            # Clear existing table data
            if dpg.does_item_exist("dmx_table"):
                children = dpg.get_item_children("dmx_table", slot=1)
                for child in children:
                    dpg.delete_item(child)
            
            # Find last non-zero channel for optimization
            last_nonzero = 512
            for i in range(511, -1, -1):
                if display_data[i] != 0:
                    last_nonzero = i + 1
                    break
            
            # Show active channels + some padding, but limit display for performance
            display_limit = min(max(last_nonzero + 6, 30), 150)  # Show at least 30, max 150 for performance
            
            # Determine channels per pixel based on master setting
            channels_per_pixel = 4 if self.master_enabled else 3
            
            # Add data rows
            for i in range(0, display_limit, channels_per_pixel):
                if self.master_enabled and i + 3 < len(display_data):
                    # RGBM format
                    r, g, b, m = display_data[i], display_data[i+1], display_data[i+2], display_data[i+3]
                    
                    # Determine if this is active data or padding
                    is_active = i < len(dmx_data)
                    pixel_num = (i // 4) + 1
                    
                    with dpg.table_row(parent="dmx_table"):
                        # Channel numbers
                        channel_text = f"{i+1}-{i+4}"
                        if not is_active:
                            channel_text += " (pad)"
                        dpg.add_text(channel_text)
                        
                        # Values
                        dpg.add_text(f"{r},{g},{b},{m}")
                        
                        # RGBM display
                        dpg.add_text(f"#{r:02x}{g:02x}{b:02x} M{m}")
                        
                        # Color preview
                        if max(r, g, b) > 0:
                            if max(r, g, b) > 128:
                                color_text = "██"  # Bright
                            else:
                                color_text = "▓▓"  # Dim
                        else:
                            color_text = "░░"  # Off/Zero
                            
                        dpg.add_text(color_text)
                
                elif not self.master_enabled and i + 2 < len(display_data):
                    # RGB format
                    r, g, b = display_data[i], display_data[i+1], display_data[i+2]
                    
                    # Determine if this is active data or padding
                    is_active = i < len(dmx_data)
                    pixel_num = (i // 3) + 1
                    
                    with dpg.table_row(parent="dmx_table"):
                        # Channel numbers
                        channel_text = f"{i+1}-{i+3}"
                        if not is_active:
                            channel_text += " (pad)"
                        dpg.add_text(channel_text)
                        
                        # Values
                        dpg.add_text(f"{r},{g},{b}")
                        
                        # RGB display
                        dpg.add_text(f"#{r:02x}{g:02x}{b:02x}")
                        
                        # Color preview
                        if max(r, g, b) > 0:
                            if max(r, g, b) > 128:
                                color_text = "██"  # Bright
                            else:
                                color_text = "▓▓"  # Dim
                        else:
                            color_text = "░░"  # Off/Zero
                            
                        dpg.add_text(color_text)
            
            # Update status
            active_channels = min(len(dmx_data), 512)
            total_channels = 512
            pixel_count = active_channels // channels_per_pixel
            
            if display_limit < 512:
                status_text = f"Total: {total_channels} channels (showing {display_limit}, active: {active_channels})"
            else:
                status_text = f"Total: {total_channels} channels (active: {active_channels})"
                
            dpg.set_value("total_channels", status_text)
            
            # Update format display
            format_text = "RGBM" if self.master_enabled else "RGB"
            dpg.set_value("channel_format", f"Format: {format_text}")
            
        except Exception as e:
            print(f"DMX table update error: {e}")
            dpg.set_value("total_channels", "Error updating table")
    
    # Zoom and navigation methods remain the same...
    def zoom_in(self):
        """Zoom in the video preview"""
        self.zoom_level = min(self.zoom_level * 1.5, 16.0)
        self.update_zoom_ui()
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def zoom_out(self):
        """Zoom out the video preview"""
        self.zoom_level = max(self.zoom_level / 1.5, 0.25)
        self.update_zoom_ui()
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def reset_zoom(self):
        """Reset zoom to 1:1"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_zoom_ui()
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def fit_to_window(self):
        """Fit video to window size"""
        if not self.video_processor.cap:
            return
        
        video_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate zoom to fit in 320x240 preview
        zoom_x = 320 / video_width
        zoom_y = 240 / video_height
        self.zoom_level = min(zoom_x, zoom_y)
        self.pan_x = 0
        self.pan_y = 0
        self.update_zoom_ui()
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def set_zoom_level(self, sender, value):
        """Set zoom level from slider"""
        self.zoom_level = value
        dpg.set_value("zoom_text", f"{value:.1f}x")
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def update_zoom_ui(self):
        """Update zoom UI elements"""
        if dpg.does_item_exist("zoom_slider"):
            dpg.set_value("zoom_slider", self.zoom_level)
        if dpg.does_item_exist("zoom_text"):
            dpg.set_value("zoom_text", f"{self.zoom_level:.1f}x")
        if dpg.does_item_exist("view_area"):
            dpg.set_value("view_area", f"View Area: {self.pan_x:.0f},{self.pan_y:.0f}")
    
    def pan_up(self):
        """Pan view up"""
        self.pan_y -= 20
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
        self.update_zoom_ui()
    
    def pan_down(self):
        """Pan view down"""
        self.pan_y += 20
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
        self.update_zoom_ui()
    
    def pan_left(self):
        """Pan view left"""
        self.pan_x -= 20
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
        self.update_zoom_ui()
    
    def pan_right(self):
        """Pan view right"""
        self.pan_x += 20
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
        self.update_zoom_ui()
    
    def center_view(self):
        """Center the view"""
        self.pan_x = 0
        self.pan_y = 0
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
        self.update_zoom_ui()
    
    def mouse_wheel_callback(self, sender, delta):
        """Handle mouse wheel for zooming"""
        if dpg.is_item_hovered("video_preview"):
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
    
    def mouse_click_callback(self, sender, button):
        """Handle mouse clicks on video preview"""
        if dpg.is_item_hovered("video_preview") and button == 0:  # Left click
            if (self.mapping_mode and 
                self.video_processor.frame_data is not None and
                hasattr(self.video_processor.frame_data, 'shape') and 
                self.video_processor.frame_data.size > 0):
                try:
                    # Initialize click counter for debug
                    if not hasattr(self, '_click_debug_count'):
                        self._click_debug_count = 0
                    self._click_debug_count += 1
                    
                    # Get mouse and image positions
                    mouse_pos = dpg.get_mouse_pos()
                    image_pos = dpg.get_item_pos("video_preview")
                    
                    # Calculate relative position within the 320x240 preview image
                    rel_x = mouse_pos[0] - image_pos[0]
                    rel_y = mouse_pos[1] - image_pos[1]
                    
                    # Ensure click is within preview bounds
                    if rel_x < 0 or rel_x >= 320 or rel_y < 0 or rel_y >= 240:
                        print(f"Click outside preview bounds: ({rel_x:.1f}, {rel_y:.1f})")
                        return
                    
                    # Get original video dimensions
                    orig_height, orig_width = self.video_processor.frame_data.shape[:2]
                    
                    # Convert preview coordinates to original video coordinates
                    # Account for zoom and pan transformations
                    
                    # Step 1: Add pan offset to get zoomed frame coordinates
                    zoomed_x = rel_x + self.pan_x
                    zoomed_y = rel_y + self.pan_y
                    
                    # Step 2: Convert from zoomed coordinates to original video coordinates
                    video_x = zoomed_x / self.zoom_level
                    video_y = zoomed_y / self.zoom_level
                    
                    # Step 3: Convert to integer pixel coordinates
                    pixel_x = int(video_x)
                    pixel_y = int(video_y)
                    
                    # Debug output (only for first few clicks)
                    debug_enabled = (hasattr(self, '_debug_pixel_extraction') and 
                                   self._debug_pixel_extraction and 
                                   self._click_debug_count <= 5)
                    
                    if debug_enabled:
                        print(f"\n=== Mouse Click Debug #{self._click_debug_count} ===")
                        print(f"Mouse pos: ({mouse_pos[0]:.1f}, {mouse_pos[1]:.1f})")
                        print(f"Image pos: ({image_pos[0]:.1f}, {image_pos[1]:.1f})")
                        print(f"Relative: ({rel_x:.1f}, {rel_y:.1f})")
                        print(f"Pan: ({self.pan_x:.1f}, {self.pan_y:.1f})")
                        print(f"Zoom: {self.zoom_level:.2f}")
                        print(f"Zoomed coords: ({zoomed_x:.1f}, {zoomed_y:.1f})")
                        print(f"Video coords: ({video_x:.1f}, {video_y:.1f})")
                        print(f"Pixel coords: ({pixel_x}, {pixel_y})")
                        print(f"Video bounds: {orig_width}x{orig_height}")
                    
                    # Check if coordinates are within video bounds
                    if 0 <= pixel_x < orig_width and 0 <= pixel_y < orig_height:
                        if debug_enabled:
                            print(f"✓ Valid pixel selected: ({pixel_x}, {pixel_y})")
                        self.select_pixel(pixel_x, pixel_y)
                    else:
                        print(f"✗ Pixel out of bounds: ({pixel_x}, {pixel_y})")
                        dpg.set_value("status_text", f"Click out of bounds: ({pixel_x}, {pixel_y})")
                    
                    if debug_enabled:
                        print("=== End Debug ===\n")
                    
                except Exception as e:
                    print(f"Mouse click error: {e}")
                    import traceback
                    traceback.print_exc()
    
    def mouse_drag_callback(self, sender, delta):
        """Handle mouse drag for panning"""
        if dpg.is_item_hovered("video_preview") and dpg.is_mouse_button_down(0):
            self.pan_x -= delta[0]
            self.pan_y -= delta[1]
            if (self.video_processor.frame_data is not None and 
                hasattr(self.video_processor.frame_data, 'shape') and 
                self.video_processor.frame_data.size > 0):
                self.update_preview(self.video_processor.frame_data)
            self.update_zoom_ui()
    
    def select_pixel(self, x: int, y: int):
        """Select a pixel for mapping - updated for master channel"""
        try:
            self.selected_pixel = (x, y)
            
            # Initialize selection counter for debug
            if not hasattr(self, '_selection_debug_count'):
                self._selection_debug_count = 0
            self._selection_debug_count += 1
            
            debug_enabled = (hasattr(self, '_debug_pixel_extraction') and 
                           self._debug_pixel_extraction and 
                           self._selection_debug_count <= 5)
            
            if debug_enabled:
                print(f"\n=== Pixel Selection #{self._selection_debug_count} ===")
                print(f"Selected pixel: ({x}, {y})")
            
            # Update UI
            dpg.set_value("selected_pixel_info", f"Pixel: ({x}, {y})")
            
            # Get current RGB value if frame is available
            if (self.video_processor.frame_data is not None and 
                hasattr(self.video_processor.frame_data, 'shape') and 
                self.video_processor.frame_data.size > 0):
                
                frame = self.video_processor.frame_data
                orig_height, orig_width = frame.shape[:2]
                
                # Ensure coordinates are valid
                if 0 <= x < orig_width and 0 <= y < orig_height:
                    # Get BGR values from OpenCV frame
                    b, g, r = frame[y, x]  # OpenCV uses BGR, note y,x order for array indexing
                    
                    # Calculate master value
                    master = self.calculate_master_value(r, g, b) if self.master_enabled else 0
                    
                    if debug_enabled:
                        print(f"Frame dimensions: {orig_width}x{orig_height}")
                        print(f"Raw BGR from frame[{y},{x}]: B={b}, G={g}, R={r}")
                        print(f"Converted RGB: ({r}, {g}, {b})")
                        if self.master_enabled:
                            print(f"Master value: {master}")
                    
                    # Update UI with RGB values
                    dpg.set_value("selected_pixel_rgb", f"RGB: ({r}, {g}, {b})")
                    dpg.set_value("pixel_rgb", f"RGB: ({r}, {g}, {b})")
                    dpg.set_value("pixel_position", f"Position: ({x}, {y})")
                    
                    # Update master value display
                    if self.master_enabled:
                        dpg.set_value("selected_pixel_master", f"Master: {master}")
                        dpg.set_value("pixel_master", f"Master: {master}")
                    else:
                        dpg.set_value("selected_pixel_master", "Master: Disabled")
                        dpg.set_value("pixel_master", "Master: Disabled")
                    
                else:
                    if debug_enabled:
                        print(f"✗ Invalid coordinates: ({x},{y}) not in {orig_width}x{orig_height}")
                    dpg.set_value("selected_pixel_rgb", "RGB: Invalid coordinates")
                    dpg.set_value("selected_pixel_master", "Master: Invalid coordinates")
            else:
                if debug_enabled:
                    print("No frame data available")
                dpg.set_value("selected_pixel_rgb", "RGB: No frame data")
                dpg.set_value("selected_pixel_master", "Master: No frame data")
            
            # Load existing mapping if available
            if (x, y) in self.pixel_mappings:
                mapping = self.pixel_mappings[(x, y)]
                dpg.set_value("r_channel", mapping.get('r', 0))
                dpg.set_value("g_channel", mapping.get('g', 0))
                dpg.set_value("b_channel", mapping.get('b', 0))
                dpg.set_value("m_channel", mapping.get('m', 0))  # Master channel
                dpg.set_value("m_channel_enable", mapping.get('m', 0) > 0)
                if debug_enabled:
                    print(f"Existing mapping: R={mapping.get('r', 0)}, G={mapping.get('g', 0)}, B={mapping.get('b', 0)}, M={mapping.get('m', 0)}")
            else:
                dpg.set_value("r_channel", 0)
                dpg.set_value("g_channel", 0)
                dpg.set_value("b_channel", 0)
                dpg.set_value("m_channel", 0)
                dpg.set_value("m_channel_enable", self.master_enabled)
                if debug_enabled:
                    print("No existing mapping")
            
            if debug_enabled:
                print("=== End Selection ===\n")
            
            # Refresh preview to show selection
            if (self.video_processor.frame_data is not None and 
                hasattr(self.video_processor.frame_data, 'shape') and 
                self.video_processor.frame_data.size > 0):
                self.update_preview(self.video_processor.frame_data)
                
        except Exception as e:
            print(f"Pixel selection error: {e}")
            import traceback
            traceback.print_exc()
    
    def toggle_mapping_mode(self):
        """Toggle pixel mapping mode"""
        self.mapping_mode = dpg.get_value("mapping_mode")
        hint_text = "Click pixels to select" if self.mapping_mode else "Mapping mode disabled"
        dpg.set_value("mapping_hint", hint_text)
        
        # Refresh preview to show/hide selection indicators
        if (self.video_processor.frame_data is not None and 
            hasattr(self.video_processor.frame_data, 'shape') and 
            self.video_processor.frame_data.size > 0):
            self.update_preview(self.video_processor.frame_data)
    
    def update_pixel_mapping(self):
        """Update pixel mapping when channels change - updated for master channel"""
        if self.selected_pixel is None:
            return
        
        try:
            x, y = self.selected_pixel
            r_ch = dpg.get_value("r_channel")
            g_ch = dpg.get_value("g_channel")
            b_ch = dpg.get_value("b_channel")
            m_ch = dpg.get_value("m_channel") if dpg.get_value("m_channel_enable") else 0
            
            if r_ch > 0 or g_ch > 0 or b_ch > 0 or m_ch > 0:
                self.pixel_mappings[(x, y)] = {
                    'r': r_ch if r_ch > 0 else 0,
                    'g': g_ch if g_ch > 0 else 0,
                    'b': b_ch if b_ch > 0 else 0,
                    'm': m_ch if m_ch > 0 else 0
                }
            else:
                # Remove mapping if all channels are 0
                if (x, y) in self.pixel_mappings:
                    del self.pixel_mappings[(x, y)]
            
            self.update_mapping_table()
        except Exception as e:
            print(f"Pixel mapping update error: {e}")
    
    def auto_assign_channel(self, color: str):
        """Auto assign next available channel - updated for master channel"""
        if self.selected_pixel is None:
            return
        
        try:
            # Find the highest used channel
            max_channel = 0
            for mapping in self.pixel_mappings.values():
                for ch in mapping.values():
                    if ch > max_channel:
                        max_channel = ch
            
            next_channel = max_channel + 1
            dpg.set_value(f"{color}_channel", next_channel)
            
            # Enable master channel checkbox if assigning master
            if color == 'm':
                dpg.set_value("m_channel_enable", True)
            
            self.update_pixel_mapping()
        except Exception as e:
            print(f"Auto channel assignment error: {e}")
    
    def apply_pixel_mapping(self):
        """Apply the current pixel mapping"""
        self.update_pixel_mapping()
        dpg.set_value("status_text", f"Mapping applied for pixel {self.selected_pixel}")
    
    def toggle_pixel_debug(self):
        """Toggle pixel extraction debug mode"""
        if hasattr(self, '_debug_pixel_extraction'):
            self._debug_pixel_extraction = not self._debug_pixel_extraction
        else:
            self._debug_pixel_extraction = True
        
        # Reset debug counters when toggling
        if self._debug_pixel_extraction:
            self._debug_custom_mapping_count = 0
            self._click_debug_count = 0
            self._selection_debug_count = 0
        
        status = "enabled" if self._debug_pixel_extraction else "disabled"
        dpg.set_value("status_text", f"Pixel extraction debug: {status}")
        print(f"Pixel extraction debug: {status}")
    
    def test_pixel_selection(self):
        """Test pixel selection accuracy"""
        if not self.video_processor.frame_data:
            dpg.set_value("status_text", "No video loaded for testing")
            return
        
        # Enable debug modes
        self._debug_pixel_extraction = True
        self.artnet_controller.enable_debug(True)
        
        print("\n=== Pixel Selection Test ===")
        print("Click on different parts of the video to test accuracy")
        print("Check console output for coordinate mapping details")
        print("Expected: clicked pixel should match displayed coordinates")
        print("=============================\n")
        
        dpg.set_value("status_text", "Pixel selection test mode - check console for details")
    
    def clear_selected_mapping(self):
        """Clear mapping for selected pixel"""
        if self.selected_pixel in self.pixel_mappings:
            del self.pixel_mappings[self.selected_pixel]
            dpg.set_value("r_channel", 0)
            dpg.set_value("g_channel", 0)
            dpg.set_value("b_channel", 0)
            dpg.set_value("m_channel", 0)
            dpg.set_value("m_channel_enable", False)
            self.update_mapping_table()
            dpg.set_value("status_text", f"Cleared mapping for pixel {self.selected_pixel}")
    
    def update_mapping_table(self):
        """Update the mapping table display - Fixed button callbacks"""
        if not dpg.does_item_exist("mapping_table"):
            return
        
        try:
            # Clear existing rows
            children = dpg.get_item_children("mapping_table", slot=1)
            for child in children:
                dpg.delete_item(child)
            
            # Add new rows with working buttons
            for (x, y), mapping in self.pixel_mappings.items():
                with dpg.table_row(parent="mapping_table"):
                    dpg.add_text(f"({x},{y})")
                    dpg.add_text(str(mapping.get('r', 0)))
                    dpg.add_text(str(mapping.get('g', 0)))
                    dpg.add_text(str(mapping.get('b', 0)))
                    dpg.add_text(str(mapping.get('m', 0)))  # Master channel
                    with dpg.group(horizontal=True):
                        # Create unique button IDs and use tag-based callbacks
                        edit_btn_id = f"edit_btn_{x}_{y}"
                        del_btn_id = f"del_btn_{x}_{y}"
                        
                        dpg.add_button(label="Edit", width=30, tag=edit_btn_id, 
                                     user_data=(x, y), callback=self._on_edit_button)
                        dpg.add_button(label="Del", width=30, tag=del_btn_id, 
                                     user_data=(x, y), callback=self._on_delete_button)
        except Exception as e:
            print(f"Mapping table update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_edit_button(self, sender, app_data, user_data):
        """Handle edit button click"""
        try:
            x, y = user_data
            print(f"Editing pixel ({x}, {y})")
            self.select_pixel(x, y)
        except Exception as e:
            print(f"Edit button error: {e}")
    
    def _on_delete_button(self, sender, app_data, user_data):
        """Handle delete button click"""
        try:
            x, y = user_data
            print(f"Deleting mapping for pixel ({x}, {y})")
            if (x, y) in self.pixel_mappings:
                del self.pixel_mappings[(x, y)]
                self.update_mapping_table()
                dpg.set_value("status_text", f"Deleted mapping for pixel ({x}, {y})")
        except Exception as e:
            print(f"Delete button error: {e}")
    
    def _edit_pixel_mapping(self, x, y, sender=None, app_data=None):
        """Helper method for editing pixel mapping - DEPRECATED"""
        pass
    
    def _delete_pixel_mapping(self, x, y, sender=None, app_data=None):
        """Helper method for deleting pixel mapping - DEPRECATED"""
        pass
    
    # Fixed channels methods - NEW
    def add_fixed_channel(self):
        """Add a new fixed channel"""
        try:
            channel = dpg.get_value("new_fixed_channel")
            value = dpg.get_value("new_fixed_value")
            name = dpg.get_value("new_fixed_name").strip()
            
            # Validation
            if channel < 1 or channel > 512:
                dpg.set_value("status_text", "Channel must be between 1 and 512")
                return
            
            if value < 0 or value > 255:
                dpg.set_value("status_text", "Value must be between 0 and 255")
                return
            
            if not name:
                name = f"Channel {channel}"
            
            # Check if channel already exists
            if channel in self.fixed_channels:
                dpg.set_value("status_text", f"Channel {channel} already exists. Update it instead.")
                return
            
            # Add the fixed channel
            self.fixed_channels[channel] = {
                'value': value,
                'name': name,
                'enabled': True
            }
            
            # Clear inputs
            dpg.set_value("new_fixed_channel", 1)
            dpg.set_value("new_fixed_value", 0)
            dpg.set_value("new_fixed_name", "")
            
            # Update table
            self.update_fixed_channels_table()
            
            dpg.set_value("status_text", f"✓ Added fixed channel {channel}: {name} = {value}")
            print(f"Added fixed channel {channel}: {name} = {value}")
            
        except Exception as e:
            error_msg = f"Failed to add fixed channel: {e}"
            dpg.set_value("status_text", error_msg)
            print(error_msg)
    
    def update_fixed_channels_table(self):
        """Update the fixed channels table display"""
        if not dpg.does_item_exist("fixed_channels_table"):
            return
        
        try:
            # Clear existing rows
            children = dpg.get_item_children("fixed_channels_table", slot=1)
            for child in children:
                dpg.delete_item(child)
            
            # Add rows for each fixed channel
            for channel in sorted(self.fixed_channels.keys()):
                info = self.fixed_channels[channel]
                
                with dpg.table_row(parent="fixed_channels_table"):
                    dpg.add_text(str(channel))
                    dpg.add_text(str(info['value']))
                    dpg.add_text(info['name'])
                    
                    # Enabled checkbox with unique tag
                    checkbox_tag = f"fixed_enabled_{channel}"
                    dpg.add_checkbox(label="", tag=checkbox_tag, default_value=info['enabled'],
                                   user_data=channel, callback=self._on_fixed_channel_toggle)
                    
                    # Action buttons
                    with dpg.group(horizontal=True):
                        edit_btn_tag = f"fixed_edit_{channel}"
                        delete_btn_tag = f"fixed_delete_{channel}"
                        
                        dpg.add_button(label="Edit", width=30, tag=edit_btn_tag,
                                     user_data=channel, callback=self._on_edit_fixed_channel)
                        dpg.add_button(label="Del", width=30, tag=delete_btn_tag,
                                     user_data=channel, callback=self._on_delete_fixed_channel)
        
        except Exception as e:
            print(f"Fixed channels table update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_fixed_channel_toggle(self, sender, value, user_data):
        """Handle fixed channel enable/disable toggle"""
        try:
            channel = user_data
            if channel in self.fixed_channels:
                self.fixed_channels[channel]['enabled'] = value
                status = "enabled" if value else "disabled"
                print(f"Fixed channel {channel} {status}")
        except Exception as e:
            print(f"Fixed channel toggle error: {e}")
    
    def _on_edit_fixed_channel(self, sender, app_data, user_data):
        """Handle edit fixed channel button"""
        try:
            channel = user_data
            if channel in self.fixed_channels:
                info = self.fixed_channels[channel]
                
                # Show edit dialog
                self._show_edit_fixed_channel_dialog(channel, info)
                
        except Exception as e:
            print(f"Edit fixed channel error: {e}")
    
    def _show_edit_fixed_channel_dialog(self, channel, info):
        """Show edit dialog for fixed channel"""
        try:
            dialog_tag = f"edit_fixed_dialog_{channel}"
            
            # Delete existing dialog if present
            if dpg.does_item_exist(dialog_tag):
                dpg.delete_item(dialog_tag)
            
            with dpg.window(label=f"Edit Fixed Channel {channel}", modal=True, show=True, 
                           tag=dialog_tag, width=350, height=200, pos=[400, 300]):
                
                dpg.add_text(f"Editing Channel {channel}")
                dpg.add_separator()
                
                # Edit fields
                value_tag = f"edit_value_{channel}"
                name_tag = f"edit_name_{channel}"
                
                dpg.add_input_int(label="Value", tag=value_tag, default_value=info['value'],
                                min_value=0, max_value=255, width=100)
                dpg.add_input_text(label="Name", tag=name_tag, default_value=info['name'], width=200)
                
                dpg.add_separator()
                
                # Buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=lambda: self._save_fixed_channel_edit(channel, value_tag, name_tag, dialog_tag))
                    dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item(dialog_tag))
                    
        except Exception as e:
            print(f"Show edit dialog error: {e}")
    
    def _save_fixed_channel_edit(self, channel, value_tag, name_tag, dialog_tag):
        """Save edited fixed channel"""
        try:
            new_value = dpg.get_value(value_tag)
            new_name = dpg.get_value(name_tag).strip()
            
            if not new_name:
                new_name = f"Channel {channel}"
            
            # Update the fixed channel
            self.fixed_channels[channel]['value'] = new_value
            self.fixed_channels[channel]['name'] = new_name
            
            # Update table and close dialog
            self.update_fixed_channels_table()
            dpg.delete_item(dialog_tag)
            
            dpg.set_value("status_text", f"✓ Updated fixed channel {channel}: {new_name} = {new_value}")
            print(f"Updated fixed channel {channel}: {new_name} = {new_value}")
            
        except Exception as e:
            print(f"Save fixed channel edit error: {e}")
    
    def _on_delete_fixed_channel(self, sender, app_data, user_data):
        """Handle delete fixed channel button"""
        try:
            channel = user_data
            if channel in self.fixed_channels:
                name = self.fixed_channels[channel]['name']
                del self.fixed_channels[channel]
                self.update_fixed_channels_table()
                dpg.set_value("status_text", f"✓ Deleted fixed channel {channel}: {name}")
                print(f"Deleted fixed channel {channel}: {name}")
        except Exception as e:
            print(f"Delete fixed channel error: {e}")
    
    def enable_all_fixed_channels(self):
        """Enable all fixed channels"""
        try:
            enabled_count = 0
            for channel in self.fixed_channels:
                if not self.fixed_channels[channel]['enabled']:
                    self.fixed_channels[channel]['enabled'] = True
                    enabled_count += 1
            
            self.update_fixed_channels_table()
            dpg.set_value("status_text", f"✓ Enabled {enabled_count} fixed channels")
            print(f"Enabled {enabled_count} fixed channels")
            
        except Exception as e:
            print(f"Enable all fixed channels error: {e}")
    
    def disable_all_fixed_channels(self):
        """Disable all fixed channels"""
        try:
            disabled_count = 0
            for channel in self.fixed_channels:
                if self.fixed_channels[channel]['enabled']:
                    self.fixed_channels[channel]['enabled'] = False
                    disabled_count += 1
            
            self.update_fixed_channels_table()
            dpg.set_value("status_text", f"✓ Disabled {disabled_count} fixed channels")
            print(f"Disabled {disabled_count} fixed channels")
            
        except Exception as e:
            print(f"Disable all fixed channels error: {e}")
    
    def clear_all_fixed_channels(self):
        """Clear all fixed channels"""
        try:
            count = len(self.fixed_channels)
            self.fixed_channels.clear()
            self.update_fixed_channels_table()
            dpg.set_value("status_text", f"✓ Cleared {count} fixed channels")
            print(f"Cleared {count} fixed channels")
            
        except Exception as e:
            print(f"Clear all fixed channels error: {e}")
    
    def add_preset_channel(self, channel, value, name):
        """Add a preset fixed channel"""
        try:
            # Check if channel already exists
            if channel in self.fixed_channels:
                dpg.set_value("status_text", f"Channel {channel} already exists")
                return
            
            # Add the preset channel
            self.fixed_channels[channel] = {
                'value': value,
                'name': name,
                'enabled': True
            }
            
            self.update_fixed_channels_table()
            dpg.set_value("status_text", f"✓ Added preset: {name} (Ch{channel}={value})")
            print(f"Added preset: {name} (Ch{channel}={value})")
            
        except Exception as e:
            print(f"Add preset channel error: {e}")
    
    def delete_mapping(self, pixel_pos: tuple):
        """Delete a specific mapping"""
        if pixel_pos in self.pixel_mappings:
            del self.pixel_mappings[pixel_pos]
            self.update_mapping_table()
            dpg.set_value("status_text", f"Deleted mapping for pixel {pixel_pos}")
    
    def auto_map_all_pixels(self):
        """Auto map all pixels sequentially - updated for master channel"""
        if not self.video_processor.cap:
            dpg.set_value("status_text", "No video loaded")
            return
        
        try:
            orig_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            start_ch = dpg.get_value("start_channel")
            include_master = dpg.get_value("preset_include_master") if dpg.does_item_exist("preset_include_master") else self.master_enabled
            current_ch = start_ch
            
            channels_per_pixel = 4 if include_master else 3
            
            for y in range(orig_height):
                for x in range(orig_width):
                    if include_master:
                        self.pixel_mappings[(x, y)] = {
                            'r': current_ch,
                            'g': current_ch + 1,
                            'b': current_ch + 2,
                            'm': current_ch + 3
                        }
                        current_ch += 4
                    else:
                        self.pixel_mappings[(x, y)] = {
                            'r': current_ch,
                            'g': current_ch + 1,
                            'b': current_ch + 2,
                            'm': 0
                        }
                        current_ch += 3
            
            self.update_mapping_table()
            format_desc = "RGBM" if include_master else "RGB"
            dpg.set_value("status_text", f"Auto mapped {orig_width}x{orig_height} pixels ({format_desc}) starting from channel {start_ch}")
        except Exception as e:
            print(f"Auto mapping error: {e}")
            dpg.set_value("status_text", f"Auto mapping failed: {e}")
    
    def apply_sequential_mapping(self):
        """Apply sequential mapping"""
        self.auto_map_all_pixels()
    
    def apply_matrix_mapping(self):
        """Apply matrix mapping with snake pattern option - updated for master channel"""
        if not self.video_processor.cap:
            dpg.set_value("status_text", "No video loaded")
            return
        
        try:
            orig_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            start_ch = dpg.get_value("start_channel")
            snake = dpg.get_value("snake_pattern")
            reverse_rows = dpg.get_value("reverse_rows")
            include_master = dpg.get_value("preset_include_master") if dpg.does_item_exist("preset_include_master") else self.master_enabled
            
            current_ch = start_ch
            channels_per_pixel = 4 if include_master else 3
            
            for y in range(orig_height):
                if reverse_rows:
                    y_actual = orig_height - 1 - y
                else:
                    y_actual = y
                
                if snake and y % 2 == 1:
                    # Reverse direction for snake pattern
                    x_range = range(orig_width - 1, -1, -1)
                else:
                    x_range = range(orig_width)
                
                for x in x_range:
                    if include_master:
                        self.pixel_mappings[(x, y_actual)] = {
                            'r': current_ch,
                            'g': current_ch + 1,
                            'b': current_ch + 2,
                            'm': current_ch + 3
                        }
                        current_ch += 4
                    else:
                        self.pixel_mappings[(x, y_actual)] = {
                            'r': current_ch,
                            'g': current_ch + 1,
                            'b': current_ch + 2,
                            'm': 0
                        }
                        current_ch += 3
            
            self.update_mapping_table()
            pattern_desc = "snake" if snake else "linear"
            format_desc = "RGBM" if include_master else "RGB"
            dpg.set_value("status_text", f"Applied {pattern_desc} matrix mapping ({format_desc})")
        except Exception as e:
            print(f"Matrix mapping error: {e}")
            dpg.set_value("status_text", f"Matrix mapping failed: {e}")
    
    def preview_matrix_mapping(self):
        """Preview matrix mapping pattern"""
        dpg.set_value("status_text", "Matrix mapping preview - check DMX Data window for pattern")
    
    def clear_all_mappings(self):
        """Clear all pixel mappings"""
        self.pixel_mappings.clear()
        self.selected_pixel = None
        dpg.set_value("selected_pixel_info", "Pixel: None selected")
        dpg.set_value("selected_pixel_rgb", "RGB: N/A")
        dpg.set_value("selected_pixel_master", "Master: N/A")
        dpg.set_value("r_channel", 0)
        dpg.set_value("g_channel", 0)
        dpg.set_value("b_channel", 0)
        dpg.set_value("m_channel", 0)
        dpg.set_value("m_channel_enable", False)
        self.update_mapping_table()
        dpg.set_value("status_text", "Cleared all mappings")
    
    def export_mappings(self):
        """Export mappings to JSON with metadata"""
        try:
            print("Starting JSON export...")
            
            # Prepare mapping data
            mapping_data = {}
            for (x, y), mapping in self.pixel_mappings.items():
                mapping_data[f"{x},{y}"] = {
                    'r': mapping.get('r', 0),
                    'g': mapping.get('g', 0),
                    'b': mapping.get('b', 0),
                    'm': mapping.get('m', 0)
                }
            
            # Prepare fixed channels data
            fixed_channels_data = {}
            for channel, info in self.fixed_channels.items():
                fixed_channels_data[str(channel)] = {
                    'value': info['value'],
                    'name': info['name'],
                    'enabled': info['enabled']
                }
            
            print(f"Prepared {len(mapping_data)} pixel mappings and {len(fixed_channels_data)} fixed channels")
            
            # Get current timestamp
            import datetime
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            readable_timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare metadata
            metadata = {
                'version': '3.0',
                'format': 'RGBM' if self.master_enabled else 'RGB',
                'master_enabled': self.master_enabled,
                'total_mappings': len(self.pixel_mappings),
                'total_fixed_channels': len(self.fixed_channels),
                'export_timestamp': readable_timestamp,
                'grid_settings': {
                    'width': dpg.get_value("grid_width") if dpg.does_item_exist("grid_width") else 16,
                    'height': dpg.get_value("grid_height") if dpg.does_item_exist("grid_height") else 16
                }
            }
            
            # Add video information if available
            if self.video_processor.cap:
                try:
                    metadata['video_info'] = {
                        'width': int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': float(self.video_processor.fps),
                        'duration': float(self.video_processor.duration),
                        'frame_count': int(self.video_processor.frame_count)
                    }
                    print("Added video info to metadata")
                except Exception as e:
                    print(f"Could not add video info: {e}")
            
            # Add master calculation method if enabled
            if self.master_enabled:
                try:
                    metadata['master_settings'] = {
                        'calculation_method': dpg.get_value("master_method") if dpg.does_item_exist("master_method") else "Luminance",
                        'custom_formula': dpg.get_value("custom_master_formula") if dpg.does_item_exist("custom_master_formula") else ""
                    }
                    print("Added master settings to metadata")
                except Exception as e:
                    print(f"Could not add master settings: {e}")
            
            # Create final export structure
            export_data = {
                'metadata': metadata,
                'pixel_mappings': mapping_data,
                'fixed_channels': fixed_channels_data
            }
            
            # Generate filename
            filename = f"pixel_mappings_{timestamp}.json"
            
            print(f"Writing to file: {filename}")
            
            # Save to JSON file
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Verify file was created
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                success_msg = f"✓ Exported {len(self.pixel_mappings)} mappings to {filename} ({file_size} bytes)"
                dpg.set_value("status_text", success_msg)
                print(success_msg)
            else:
                raise Exception("File was not created")
            
        except Exception as e:
            error_msg = f"Export failed: {e}"
            dpg.set_value("status_text", error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
    
    def import_mappings(self):
        """Import mappings from JSON with validation"""
        try:
            print("Opening import dialog...")
            
            # Create file dialog for JSON import
            if not dpg.does_item_exist("import_dialog"):
                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=self._import_mappings_callback,
                    tag="import_dialog",
                    width=600,
                    height=400
                ):
                    dpg.add_file_extension("JSON Files{.json}", color=(0, 255, 0, 255))
                    dpg.add_file_extension(".*")
            
            dpg.show_item("import_dialog")
            
        except Exception as e:
            error_msg = f"Import dialog error: {e}"
            dpg.set_value("status_text", error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
    
    def _import_mappings_callback(self, sender, app_data):
        """Callback for import file dialog"""
        try:
            file_path = app_data['file_path_name']
            print(f"Importing from: {file_path}")
            
            # Read and parse JSON file
            with open(file_path, "r", encoding='utf-8') as f:
                import_data = json.load(f)
            
            print("JSON file loaded successfully")
            
            # Validate file structure
            if 'metadata' not in import_data or 'pixel_mappings' not in import_data:
                raise ValueError("Invalid file format: missing metadata or pixel_mappings")
            
            metadata = import_data['metadata']
            pixel_mappings = import_data['pixel_mappings']
            fixed_channels_data = import_data.get('fixed_channels', {})  # Optional for backward compatibility
            
            # Display import information
            print(f"\n=== Importing Mapping Data ===")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Version: {metadata.get('version', 'Unknown')}")
            print(f"Format: {metadata.get('format', 'Unknown')}")
            print(f"Total mappings: {metadata.get('total_mappings', 0)}")
            print(f"Total fixed channels: {metadata.get('total_fixed_channels', 0)}")
            print(f"Export date: {metadata.get('export_timestamp', 'Unknown')}")
            
            if 'video_info' in metadata:
                video_info = metadata['video_info']
                print(f"Original video: {video_info.get('width', 0)}x{video_info.get('height', 0)} @ {video_info.get('fps', 0):.1f}fps")
            
            # Warn about format compatibility
            imported_master_enabled = metadata.get('master_enabled', False)
            if imported_master_enabled != self.master_enabled:
                warning = f"⚠️  Format mismatch: imported data is {metadata.get('format', 'Unknown')}, current setting is {'RGBM' if self.master_enabled else 'RGB'}"
                print(warning)
                dpg.set_value("status_text", warning)
            
            # Clear existing mappings and fixed channels
            self.pixel_mappings.clear()
            self.fixed_channels.clear()
            print("Cleared existing mappings and fixed channels")
            
            # Import pixel mappings
            imported_count = 0
            for coord_str, mapping in pixel_mappings.items():
                try:
                    x_str, y_str = coord_str.split(',')
                    x, y = int(x_str), int(y_str)
                    
                    self.pixel_mappings[(x, y)] = {
                        'r': mapping.get('r', 0),
                        'g': mapping.get('g', 0),
                        'b': mapping.get('b', 0),
                        'm': mapping.get('m', 0)
                    }
                    imported_count += 1
                    
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid mapping entry: {coord_str} - {e}")
                    continue
            
            print(f"Imported {imported_count} pixel mappings")
            
            # Import fixed channels
            fixed_imported_count = 0
            for channel_str, info in fixed_channels_data.items():
                try:
                    channel = int(channel_str)
                    if 1 <= channel <= 512:
                        self.fixed_channels[channel] = {
                            'value': info.get('value', 0),
                            'name': info.get('name', f'Channel {channel}'),
                            'enabled': info.get('enabled', True)
                        }
                        fixed_imported_count += 1
                    else:
                        print(f"Skipping invalid channel number: {channel}")
                        
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid fixed channel entry: {channel_str} - {e}")
                    continue
            
            print(f"Imported {fixed_imported_count} fixed channels")
            
            # Update UI
            self.update_mapping_table()
            self.update_fixed_channels_table()
            
            # Update master settings if available and compatible
            if imported_master_enabled and 'master_settings' in metadata:
                master_settings = metadata['master_settings']
                method = master_settings.get('calculation_method', '')
                
                if dpg.does_item_exist("master_method") and method:
                    try:
                        dpg.set_value("master_method", method)
                        if method == "Custom" and dpg.does_item_exist("custom_master_formula"):
                            formula = master_settings.get('custom_formula', '')
                            if formula:
                                dpg.set_value("custom_master_formula", formula)
                        print(f"Restored master method: {method}")
                    except Exception as e:
                        print(f"Could not set master method: {method} - {e}")
            
            # Update grid settings if available
            if 'grid_settings' in metadata:
                grid_settings = metadata['grid_settings']
                width = grid_settings.get('width', 16)
                height = grid_settings.get('height', 16)
                
                try:
                    if dpg.does_item_exist("grid_width"):
                        dpg.set_value("grid_width", width)
                    if dpg.does_item_exist("grid_height"):
                        dpg.set_value("grid_height", height)
                    if dpg.does_item_exist("grid_width_main"):
                        dpg.set_value("grid_width_main", width)
                    if dpg.does_item_exist("grid_height_main"):
                        dpg.set_value("grid_height_main", height)
                    print(f"Restored grid settings: {width}x{height}")
                except Exception as e:
                    print(f"Could not restore grid settings: {e}")
            
            success_msg = f"✓ Imported {imported_count} mappings and {fixed_imported_count} fixed channels from {os.path.basename(file_path)}"
            dpg.set_value("status_text", success_msg)
            print(success_msg)
            print("=== Import Complete ===\n")
            
        except Exception as e:
            error_msg = f"Import failed: {e}"
            dpg.set_value("status_text", error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
    
    def update_grid_display(self):
        """Update grid display setting"""
        if dpg.does_item_exist("show_grid"):
            dpg.set_value("show_grid", dpg.get_value("show_grid_preview"))
        # Refresh current frame if video is loaded
        if self.video_processor.frame_data is not None:
            self.update_preview(self.video_processor.frame_data)
    
    def sync_grid_settings(self):
        """Sync grid settings between windows"""
        if dpg.does_item_exist("show_grid_preview"):
            dpg.set_value("show_grid_preview", dpg.get_value("show_grid"))
        # Refresh current frame if video is loaded
        if self.video_processor.frame_data is not None:
            self.update_preview(self.video_processor.frame_data)
    
    def set_pixel_perfect_view(self):
        """Set pixel perfect view mode"""
        if not self.video_processor.cap:
            return
        
        # Set zoom to integer multiple for pixel perfect display
        video_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Find the largest integer zoom that fits in preview
        max_zoom_x = 320 // video_width
        max_zoom_y = 240 // video_height
        zoom = max(1, min(max_zoom_x, max_zoom_y))
        
        self.zoom_level = float(zoom)
        self.pan_x = 0
        self.pan_y = 0
        self.update_zoom_ui()
        
        if self.video_processor.frame_data is not None:
            self.update_preview(self.video_processor.frame_data)
        
        dpg.set_value("status_text", f"Pixel perfect view enabled ({zoom}x)")
    
    def update_grid_settings(self):
        """Update grid settings from main window"""
        if dpg.does_item_exist("grid_width"):
            dpg.set_value("grid_width", dpg.get_value("grid_width_main"))
        if dpg.does_item_exist("grid_height"):
            dpg.set_value("grid_height", dpg.get_value("grid_height_main"))
    
    def update_brightness(self):
        """Update brightness from main window"""
        if dpg.does_item_exist("brightness"):
            dpg.set_value("brightness", dpg.get_value("brightness_main"))
    
    def set_grid_preset(self, width: int, height: int):
        """Set grid preset"""
        dpg.set_value("grid_width", width)
        dpg.set_value("grid_height", height)
        dpg.set_value("grid_width_main", width)
        dpg.set_value("grid_height_main", height)
    
    def reset_preview_size(self):
        """Reset preview to original size"""
        dpg.set_value("status_text", "Preview reset to original size")
    
    def export_dmx_csv(self):
        """Export current DMX data to JSON format"""
        if not self.current_dmx_data:
            dpg.set_value("status_text", "No DMX data to export")
            return
        
        try:
            import datetime
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            readable_timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            
            channels_per_pixel = 4 if self.master_enabled else 3
            pixel_count = len(self.current_dmx_data) // channels_per_pixel
            
            # Prepare DMX data structure
            dmx_export = {
                'metadata': {
                    'version': '3.0',
                    'format': 'RGBM' if self.master_enabled else 'RGB',
                    'channels_per_pixel': channels_per_pixel,
                    'total_channels': len(self.current_dmx_data),
                    'pixel_count': pixel_count,
                    'export_timestamp': readable_timestamp,
                    'frame_info': {
                        'current_frame': self.video_processor.current_frame,
                        'total_frames': self.video_processor.frame_count,
                        'current_time': self.video_processor.get_current_time()
                    }
                },
                'dmx_data': []
            }
            
            # Convert DMX data to structured format
            for i in range(0, len(self.current_dmx_data), channels_per_pixel):
                if self.master_enabled and i + 3 < len(self.current_dmx_data):
                    r, g, b, m = self.current_dmx_data[i:i+4]
                    pixel_data = {
                        'pixel': i // 4,
                        'channels': {
                            'start': i + 1,  # 1-based channel numbering
                            'r': r,
                            'g': g,
                            'b': b,
                            'm': m
                        }
                    }
                elif not self.master_enabled and i + 2 < len(self.current_dmx_data):
                    r, g, b = self.current_dmx_data[i:i+3]
                    pixel_data = {
                        'pixel': i // 3,
                        'channels': {
                            'start': i + 1,  # 1-based channel numbering
                            'r': r,
                            'g': g,
                            'b': b
                        }
                    }
                else:
                    continue
                
                dmx_export['dmx_data'].append(pixel_data)
            
            # Save to JSON file
            filename = f"dmx_data_{timestamp}.json"
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(dmx_export, f, indent=2, ensure_ascii=False)
            
            format_desc = "RGBM" if self.master_enabled else "RGB"
            success_msg = f"✓ DMX data ({format_desc}) exported to {filename} ({pixel_count} pixels)"
            dpg.set_value("status_text", success_msg)
            print(success_msg)
            
        except Exception as e:
            error_msg = f"DMX export failed: {e}"
            dpg.set_value("status_text", error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
    
    def clear_dmx_data(self):
        """Clear DMX data table"""
        if dpg.does_item_exist("dmx_table"):
            children = dpg.get_item_children("dmx_table", slot=1)
            for child in children:
                dpg.delete_item(child)
        dpg.set_value("total_channels", "Total Channels: 0")
        dpg.set_value("status_text", "DMX data cleared")
    
    def run(self):
        """Run application"""
        try:
            dpg.start_dearpygui()
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            # Cleanup
            if self.video_processor.cap:
                self.video_processor.cap.release()
            if self.artnet_controller.socket:
                self.artnet_controller.socket.close()
            pygame.quit()
            dpg.destroy_context()

def main():
    """Main function"""
    try:
        app = VideoToDMXApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""Generate high-resolution application icon (.ico) for SIGMA Signal Analysis."""
import os
import math
from PIL import Image, ImageDraw

def create_sigma_icon(size: int = 256) -> Image.Image:
    # Create high-res RGBA image
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background rounded rectangle with gradient effect
    padding = size // 16
    rect = [padding, padding, size - padding, size - padding]
    corner_radius = size // 5
    
    # Dark modern slate-blue background
    draw.rounded_rectangle(rect, radius=corner_radius, fill=(15, 23, 42, 255), outline=(56, 189, 248, 200), width=max(2, size // 64))
    
    # Draw RF Signal Waveform (sine wave with modulation)
    points = []
    cy = size // 2
    cx_start = padding + size // 8
    cx_end = size - padding - size // 8
    width = cx_end - cx_start
    
    for x in range(cx_start, cx_end + 1):
        t = (x - cx_start) / width
        # Modulation envelope: sin wave with amplitude modulation
        envelope = math.sin(t * math.pi)
        carrier = math.sin(t * math.pi * 8)
        y = cy - int(carrier * envelope * (size * 0.28))
        points.append((x, y))
    
    # Draw glow effect behind waveform
    for w in range(size // 16, 0, -2):
        alpha = int(40 * (1 - w / (size // 16)))
        draw.line(points, fill=(14, 165, 233, alpha), width=w, joint="curve")
        
    # Draw primary vibrant cyan line
    draw.line(points, fill=(56, 189, 248, 255), width=max(3, size // 32), joint="curve")
    
    # Draw central Σ (Sigma) symbol accent
    # Draw a subtle Greek Sigma background or overlay text
    # Draw a stylized Σ in upper right / center
    return img

def main():
    os.makedirs("assets", exist_ok=True)
    sizes = [256, 128, 64, 48, 32, 16]
    images = [create_sigma_icon(s) for s in sizes]
    
    ico_path = os.path.join("assets", "sigma.ico")
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:]
    )
    print(f"Icon generated successfully at {ico_path}")

if __name__ == "__main__":
    main()

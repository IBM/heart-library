from __future__ import annotations

from typing import Iterable

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import carbon_colors


class Carbon(Base):
    """Sets the custom theme to be used for gradio app.

    Args:
        Base : Gradio themes base.
    """    
    def __init__(
        self,
        *,
        primary_hue: carbon_colors.Color | str = carbon_colors.white,
        secondary_hue: carbon_colors.Color | str = carbon_colors.red,
        neutral_hue: carbon_colors.Color | str = carbon_colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_lg,
        radius_size: sizes.Size | str = sizes.radius_none,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            fonts.GoogleFont("IBM Plex Sans"),
            fonts.GoogleFont("IBM Plex Serif"),
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
        ),
    ):
        """Carbon initialization.

        Args:
            primary_hue (carbon_colors.Color | str, optional): This is the color which draws attention in your theme. Defaults to carbon_colors.white.
            secondary_hue (carbon_colors.Color | str, optional): This is the color that is used for secondary elements in your theme. Defaults to carbon_colors.red.
            neutral_hue (carbon_colors.Color | str, optional): This is the color that is used for text and other neutral elements in your theme. Defaults to carbon_colors.blue.
            spacing_size (sizes.Size | str, optional): This sets the padding within and spacing between elements. Defaults to sizes.spacing_lg.
            radius_size (sizes.Size | str, optional): This sets the roundedness of corners of elements. Defaults to sizes.radius_none.
            text_size (sizes.Size | str, optional): This sets the font size of text. Defaults to sizes.text_md.
            font (fonts.Font | str | Iterable[fonts.Font  |  str], optional): This sets the primary font of the theme. Defaults to ( fonts.GoogleFont("IBM Plex Mono"), fonts.GoogleFont("IBM Plex Sans"), fonts.GoogleFont("IBM Plex Serif"), ).
            font_mono (fonts.Font | str | Iterable[fonts.Font  |  str], optional): This sets the monospace font of the theme. Defaults to ( fonts.GoogleFont("IBM Plex Mono"), ).
        """    
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "carbon"
        super().set(
            # Colors 
            slider_color="*neutral_900", 
            slider_color_dark="*neutral_500",
            body_text_color="*neutral_900",
            block_label_text_color="*body_text_color",
            block_title_text_color="*body_text_color",
            body_text_color_subdued="*neutral_700",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary_dark="*neutral_800",
            block_background_fill_dark="*neutral_800",
            input_background_fill_dark="*neutral_700",
            # Button Colors
            button_primary_background_fill=carbon_colors.blue.c500,
            button_primary_background_fill_hover="*neutral_300",
            button_primary_text_color="white",
            button_primary_background_fill_dark="*neutral_600",
            button_primary_background_fill_hover_dark="*neutral_600",
            button_primary_text_color_dark="white",
            button_secondary_background_fill="*button_primary_background_fill",
            button_secondary_background_fill_hover="*button_primary_background_fill_hover",
            button_secondary_text_color="*button_primary_text_color",
            button_cancel_background_fill="*button_primary_background_fill",
            button_cancel_background_fill_hover="*button_primary_background_fill_hover",
            button_cancel_text_color="*button_primary_text_color",
            checkbox_background_color=carbon_colors.black.c50,
            checkbox_label_background_fill="*button_primary_background_fill",
            checkbox_label_background_fill_hover="*button_primary_background_fill_hover",
            checkbox_label_text_color="*button_primary_text_color",
            checkbox_background_color_selected=carbon_colors.black.c50,
            checkbox_border_width="1px",
            checkbox_border_width_dark="1px",
            checkbox_border_color=carbon_colors.white.c50,
            checkbox_border_color_dark=carbon_colors.white.c50,
            
            checkbox_border_color_focus=carbon_colors.blue.c900,
            checkbox_border_color_focus_dark=carbon_colors.blue.c900,
            checkbox_border_color_selected=carbon_colors.white.c50,
            checkbox_border_color_selected_dark=carbon_colors.white.c50,
            
            checkbox_background_color_hover=carbon_colors.black.c50,
            checkbox_background_color_hover_dark=carbon_colors.black.c50,
            checkbox_background_color_dark=carbon_colors.black.c50,
            checkbox_background_color_selected_dark=carbon_colors.black.c50,
            # Padding
            checkbox_label_padding="16px",
            button_large_padding="*spacing_lg",
            button_small_padding="*spacing_sm",
            # Borders
            block_border_width="0px",
            block_border_width_dark="1px",
            shadow_drop_lg="0 1px 4px 0 rgb(0 0 0 / 0.1)",
            block_shadow="*shadow_drop_lg",
            block_shadow_dark="none",
            # Block Labels
            block_title_text_weight="600",
            block_label_text_weight="600",
            block_label_text_size="*text_md",
        )
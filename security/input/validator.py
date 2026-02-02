"""
Pydantic schema validation for travel planner inputs and outputs.
"""

from datetime import date, datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class TravelRequestInput(BaseModel):
    """Schema for validating user travel request input."""

    user_input: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="User's travel planning request"
    )

    @field_validator('user_input')
    @classmethod
    def validate_user_input(cls, v: str) -> str:
        # Check for null bytes
        if '\x00' in v:
            raise ValueError("Input contains invalid characters")
        return v.strip()


class ExtractedTravelInfo(BaseModel):
    """Schema for validating extracted travel information from LLM."""

    departure_city: str = Field(
        default="",
        max_length=100,
        description="City of departure"
    )
    destination_country: str = Field(
        default="",
        max_length=100,
        description="Destination country"
    )
    preferences: str = Field(
        default="",
        max_length=500,
        description="Travel preferences"
    )
    budget: str = Field(
        default="",
        max_length=100,
        description="Budget range"
    )
    outbound_date: str = Field(
        default="",
        max_length=20,
        description="Departure date"
    )
    return_date: str = Field(
        default="",
        max_length=20,
        description="Return date"
    )

    @field_validator('departure_city', 'destination_country')
    @classmethod
    def validate_location(cls, v: str) -> str:
        if v:
            # Only allow alphanumeric, spaces, hyphens, apostrophes
            if not re.match(r"^[\w\s\-',.]+$", v, re.UNICODE):
                raise ValueError(f"Invalid location format: {v}")
        return v.strip()

    @field_validator('outbound_date', 'return_date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        if v:
            # Try to parse common date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                try:
                    datetime.strptime(v, fmt)
                    return v
                except ValueError:
                    continue
            # If no format matched but looks like a date, allow it
            if not re.match(r'^[\d\-/]+$', v):
                raise ValueError(f"Invalid date format: {v}")
        return v

    @model_validator(mode='after')
    def validate_dates_order(self):
        """Ensure return date is after outbound date."""
        if self.outbound_date and self.return_date:
            try:
                outbound = datetime.strptime(self.outbound_date, '%Y-%m-%d')
                return_dt = datetime.strptime(self.return_date, '%Y-%m-%d')
                if return_dt < outbound:
                    raise ValueError("Return date must be after outbound date")
            except ValueError as e:
                if "Return date" in str(e):
                    raise
                # Parsing failed, skip validation
                pass
        return self


class RefinedTravelInfo(BaseModel):
    """Schema for validating refined travel information from LLM."""

    destination_city: str = Field(
        default="",
        max_length=100,
        description="Specific destination city"
    )
    arrival_airport: str = Field(
        default="",
        max_length=10,
        description="Airport code"
    )
    notes: str = Field(
        default="",
        max_length=1000,
        description="Additional notes"
    )

    @field_validator('arrival_airport')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        if v:
            # Airport codes are typically 3-4 uppercase letters
            v = v.upper().strip()
            if not re.match(r'^[A-Z]{3,4}$', v):
                # Allow longer strings that might be airport names
                if len(v) > 50:
                    raise ValueError(f"Invalid airport code/name: {v}")
        return v


class FlightInfo(BaseModel):
    """Schema for validating flight information."""

    airline: str = Field(default="Unknown Airline", max_length=100)
    flight_number: str = Field(default="N/A", max_length=20)
    cabin_class: str = Field(default="N/A", max_length=50)
    aircraft: str = Field(default="N/A", max_length=100)
    price: Any = Field(default="N/A")  # Can be string or number
    departure_time: str = Field(default="N/A", max_length=50)
    arrival_time: str = Field(default="N/A", max_length=50)
    duration: Any = Field(default="N/A")  # Can be string or number
    layovers: str = Field(default="None", max_length=500)


class HotelInfo(BaseModel):
    """Schema for validating hotel information."""

    name: str = Field(default="Unknown Hotel", max_length=200)
    stars: Any = Field(default="N/A")
    rating: Any = Field(default="N/A")
    reviews: Any = Field(default="N/A")
    check_in: str = Field(default="N/A", max_length=50)
    check_out: str = Field(default="N/A", max_length=50)
    price_per_night: Any = Field(default="N/A")
    top_amenities: str = Field(default="Not listed", max_length=500)
    website: str = Field(default="Not available", max_length=500)


class InputValidator:
    """
    Input validator using Pydantic schemas.
    """

    @staticmethod
    def validate_travel_request(user_input: str) -> tuple[bool, Optional[TravelRequestInput], Optional[str]]:
        """
        Validate user travel request input.

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        try:
            validated = TravelRequestInput(user_input=user_input)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def validate_extracted_info(data: Dict[str, Any]) -> tuple[bool, Optional[ExtractedTravelInfo], Optional[str]]:
        """
        Validate extracted travel information.

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        try:
            validated = ExtractedTravelInfo.model_validate(data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def validate_refined_info(data: Dict[str, Any]) -> tuple[bool, Optional[RefinedTravelInfo], Optional[str]]:
        """
        Validate refined travel information.

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        try:
            validated = RefinedTravelInfo.model_validate(data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def validate_flight_data(data: Dict[str, Any]) -> tuple[bool, Optional[FlightInfo], Optional[str]]:
        """
        Validate flight information.

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        try:
            validated = FlightInfo.model_validate(data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def validate_hotel_data(data: Dict[str, Any]) -> tuple[bool, Optional[HotelInfo], Optional[str]]:
        """
        Validate hotel information.

        Returns:
            Tuple of (is_valid, validated_model, error_message)
        """
        try:
            validated = HotelInfo.model_validate(data)
            return True, validated, None
        except Exception as e:
            return False, None, str(e)

import streamlit as st
import io
import re
import math
import base64
import pandas as pd
import binascii # for debug - remove
from bisect import bisect
from struct import pack, unpack


help_sensus="Copy the above Raw string, then click the link to open Sensus and paste it the 'Raw' field. In 'raw Analysis' panel, you can then click 'read raw' to get detailed analysis"
sensus="[Analyze or convert this signal in Sensus IR & RF Code Converter](https://pasthev.github.io/sensus/)"
license="This software is under GPL-3.0 license."
source="Source code is available on [GitHub](https://github.com/pasthev/irtuya)"
message="Receiving just a quick 'thank you from <*your location*>' is always a pleasure for me, and questions / feedback are always welcome:"
contact="[Contact me](https://docs.google.com/forms/d/e/1FAIpQLSckf2f04hYhTN3T6GvchbxhjhKcYHLMRDXnrRfqlM_eRW_NiA/viewform?usp=sf_link)"
credits="""<div style="text-align: center;"><a href="https://pasthev.github.io/" target="_blank"><i>pasthev 2025</i></a></div>"""
thanks="Many thanks to [mildsunrise](https://gist.github.com/mildsunrise/1d576669b63a260d2cff35fda63ec0b5) for her work on the Tuya encoding format"

RAW = "Raw"
TUYA = "Tuya"
NEC = "NEC"
BROADLINK_HEX = "Broadlink Hex"
BROADLINK_B64 = "Broadlink B64"

MAX_OMNIBOX_INPUT_LENGTH=1200       # Arbitrary limitation of 1200 caracters in omnibox, to prevent accidental disastrous pastes
MAX_NECSEQ_INPUT_LENGTH=30          # Arbitrary limitation of Nec sequence input length
MAX_NECCOM_INPUT_LENGTH=12          # Arbitrary limitation of Nec commands input length

example_sequences = {
    "Examples": [
        "",
        "Tuya: BZgjrRE/AuAXAQGDBuAFA0AB4AcTQA/gFwFAI+ALAwcQoZgj0Qg/Ag==",
        "Broadlink B64: JgBFAAABJ5MSEhISEhISNxI3EjcSNxI3EhISEhISEhISEhISEhISEhI3EjcSEhI3EhISEhI3EhISNxISEjcSNxISEjcSEhISEg0FAAAAAAA=",
        "Broadlink Hex: 2600350000012793121212121212123712371212123712121212121212371212123712121237123712121212123712121237123712121212120D050000000000",
        "Raw: 9000, 4500, 562, 1688, 562, 1688, 562, 1688, 562, 1688, 562, 1688, 562, 1688, 562, 1688, 562, 1688, 562, 563, 562, 563, 562, 563, 562, 563, 562, 563, 562, 563, 562, 563, 562, 563, 562, 563, 562, 1688, 562, 563, 562, 563, 562, 1688, 562, 563, 562, 563, 562, 563, 562, 1688, 562, 563, 562, 1688, 562, 1688, 562, 563, 562, 1688, 562, 1688, 562, 1688, 562",
        "NEC Hex: 20, DF, 10, EF",
        "NEC Binary: 00100000 11011111 00010000 11101111",
        "Brackets accepted: {([00, F7, 40, BF])}",
        "NEC Hex by pairs: 1f00 d2b4",
        "NEC Hex collated: 00EF03",
        "Nec Decimals: 32 223 16 239",
        "Nec Decimal Lists: [32, 223, 16, 239]",
        "Nec Truncated binaries: 100000, 11011111, 10000, 11101111",
        "Nec Collated binaries: 00100000110111110001000011101111"
    ]
}
example_nec_seq = {
    "Examples": [
        "",
        "20, DF, 10, EF",
        "[00, F7, 40, BF]",
        "1f00 d2b4",
        "00EF03",
        "32, 223, 16, 239",
        "[32, 223, 16, 239]",
        "0204",
        "1112",
        "00100000 11011111 00010000 11101111",
        "100000, 11011111, 10000, 11101111",
        "00100000110111110001000011101111"
    ]
}
example_nec_com = {
    "Examples": [
        "",
        "00EF03",
        "0204",
        "1b, 55",
        "[2f, 00, 02, 03]",
        "00111000 11011111 00010000"
    ]
}
#
#
# --------------------------------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------------------------------
#
# Tuya to raw
def decode_tuya_to_raw(tuya_code_string: str) -> list[int]:
    '''
    Decodes a Tuya IR code compressed string into a raw IR signal (list of durations).
    Optimized for performance using memoryview.
    '''
    payload_bytes = base64.b64decode(tuya_code_string) # Optimized base64 decoding
    payload_bytes = _decompress_fastlz(io.BytesIO(payload_bytes))

    ir_signal_durations = []
    buffer = memoryview(payload_bytes) # Use memoryview for efficient access
    for i in range(0, len(payload_bytes), 2):
        ir_signal_durations.append(unpack('<H', buffer[i:i+2])[0]) # Efficient unpacking
    return ir_signal_durations

def _decompress_fastlz(input_io: io.FileIO) -> bytes:
    ''' Reads a FastLZ compressed Tuya stream, and returns the decompressed byte string. '''
    output_bytes = bytearray()

    while (header_byte := input_io.read(1)):
        length_bits = header_byte[0] >> 5
        distance_bits = header_byte[0] & 0b11111
        if not length_bits:
            # literal block
            length_bytes = distance_bits + 1
            data_block = input_io.read(length_bytes)
            assert len(data_block) == length_bytes
        else:
            # length-distance pair block
            if length_bits == 7:
                length_bits += input_io.read(1)[0]
            length_bytes = length_bits + 2
            distance_val = (distance_bits << 8 | input_io.read(1)[0]) + 1
            data_block = bytearray()
            while len(data_block) < length_bytes:
                data_block.extend(output_bytes[-distance_val:][:length_bytes-len(data_block)])
        output_bytes.extend(data_block)

    return bytes(output_bytes)

# Raw to Tuya 
def encode_raw_to_tuya(raw_signal_durations: list[int], compression_level: int = 2) -> str:
    '''
    Encodes a raw IR signal (list of durations) into a Tuya IR code string.
    '''
    payload_bytes = b''.join(pack('<H', duration) for duration in raw_signal_durations)
    _compress_fastlz(output_io := io.BytesIO(), payload_bytes, compression_level)
    payload_bytes = output_io.getvalue()
    return base64.encodebytes(payload_bytes).decode('ascii').replace('\n', '')

def _compress_fastlz(output_io: io.FileIO, data_bytes: bytes, level: int = 2):
    '''
    Takes a byte string and outputs a FastLZ compressed "Tuya stream".
    Implemented compression levels:
    0 - copy over (no compression, 3.1% overhead)
    1 - eagerly use first length-distance pair found (linear)
    2 - eagerly use best length-distance pair found
    3 - optimal compression (n^3)
    '''
    if level == 0:
        return _emit_literal_blocks(output_io, data_bytes)

    window_size = 2**13 # window size
    max_length = 255+9 # maximum length
    distance_candidates_func = lambda pos: range(1, min(pos, window_size) + 1)

    def find_length_for_distance(start_pos: int, current_pos: int) -> int:
        length_val = 0
        limit_length = min(max_length, len(data_bytes) - current_pos)
        while length_val < limit_length and data_bytes[current_pos + length_val] == data_bytes[start_pos + length_val]:
            length_val += 1
        return length_val
    find_length_candidates_func = lambda pos: \
        ( (find_length_for_distance(pos - distance, pos), distance) for distance in distance_candidates_func(pos) )
    find_length_cheap_func = lambda pos: \
        next((candidate for candidate in find_length_candidates_func(pos) if candidate[0] >= 3), None)
    find_length_max_func = lambda pos: \
        max(find_length_candidates_func(pos), key=lambda candidate: (candidate[0], -candidate[1]), default=None)

    if level >= 2:
        suffixes_list = []; next_pos = 0
        key_func = lambda n: data_bytes[n:]
        find_index_func = lambda n: bisect(suffixes_list, key_func(n), key=key_func)
        def distance_candidates_func(pos):
            nonlocal next_pos
            while next_pos <= pos:
                if len(suffixes_list) == window_size:
                    suffixes_list.pop(find_index_func(next_pos - window_size))
                suffixes_list.insert(index_val := find_index_func(next_pos), next_pos)
                next_pos += 1
            indices = (index_val+i for i in (+1,-1)) # try +1 first
            return (pos - suffixes_list[i] for i in indices if 0 <= i < len(suffixes_list))

    if level <= 2:
        find_length_function = { 1: find_length_cheap_func, 2: find_length_max_func }[level]
        block_start_pos = current_pos = 0
        while current_pos < len(data_bytes):
            if (candidate := find_length_function(current_pos)) and candidate[0] >= 3:
                _emit_literal_blocks(output_io, data_bytes[block_start_pos:current_pos])
                _emit_distance_block(output_io, candidate[0], candidate[1])
                current_pos += candidate[0]
                block_start_pos = current_pos
            else:
                current_pos += 1
        _emit_literal_blocks(output_io, data_bytes[block_start_pos:current_pos])
        return

    # use topological sort to find shortest path
    predecessors_list = [(0, None, None)] + [None] * len(data_bytes)
    def put_edge(cost_val, length_val, distance_val):
        next_pos = current_pos + length_val
        cost_val += predecessors_list[current_pos][0]
        current_predecessor = predecessors_list[next_pos]
        if not current_predecessor or cost_val < current_predecessor[0]:
            predecessors_list[next_pos] = cost_val, length_val, distance_val
    for current_pos in range(len(data_bytes)):
        if candidate := find_length_max_func(current_pos):
            for length_val in range(3, candidate[0] + 1):
                put_edge(2 if length_val < 9 else 3, length_val, candidate[1])
        for length_val in range(1, min(32, len(data_bytes) - current_pos) + 1):
            put_edge(1 + length_val, length_val, 0)

    # reconstruct path, emit blocks
    blocks_list = []; current_pos = len(data_bytes)
    while current_pos > 0:
        _, length_val, distance_val = predecessors_list[current_pos]
        current_pos -= length_val
        blocks_list.append((current_pos, length_val, distance_val))
    for block_pos, block_length, distance_val in reversed(blocks_list):
        if not distance_val:
            _emit_literal_block(output_io, data_bytes[block_pos:block_pos + block_length])
        else:
            _emit_distance_block(output_io, block_length, distance_val)

def _emit_literal_blocks(output_io: io.FileIO, data_bytes: bytes):
    """Emits literal blocks to the output IO stream."""
    for i in range(0, len(data_bytes), 32):
        _emit_literal_block(output_io, data_bytes[i:i+32])

def _emit_literal_block(output_io: io.FileIO, data_block: bytes):
    """Emits a single literal block to the output IO stream."""
    block_length = len(data_block) - 1
    assert 0 <= block_length < (1 << 5)
    output_io.write(bytes([block_length]))
    output_io.write(data_block)

def _emit_distance_block(output_io: io.FileIO, block_length: int, distance_val: int):
    """Emits a distance block to the output IO stream."""
    distance_val -= 1
    assert 0 <= distance_val < (1 << 13)
    block_length -= 2
    assert block_length > 0
    block_bytes = bytearray()
    if block_length >= 7:
        assert block_length - 7 < (1 << 8)
        block_bytes.append(block_length - 7)
        block_length = 7
    block_bytes.insert(0, block_length << 5 | distance_val >> 8)
    block_bytes.append(distance_val & 0xFF)
    output_io.write(block_bytes)

# Raw to Nec sequence
def decode_raw_to_nec_sequence(ir_signal_durations: list[int]) -> list[int]:
    """
    Detects if a raw IR signal (list of durations) contains a valid NEC sequence (list of bytes).
    Returns the NEC sequence as a list of bytes if detected, otherwise an empty list.
    """
    nec_header_low_duration, nec_header_high_duration = 9000, 4500
    nec_bit_mark_duration, nec_zero_space_duration, nec_one_space_duration = 560, 560, 1690
    tolerance_duration = 200

    def within_tolerance(value, target):
        return target - tolerance_duration <= value <= target + tolerance_duration

    if not ir_signal_durations:
        return []

    for i in range(len(ir_signal_durations) - 1):
        if within_tolerance(ir_signal_durations[i], nec_header_low_duration) and within_tolerance(ir_signal_durations[i + 1], nec_header_high_duration):
            nec_sequence_bits = []
            j = i + 2
            while j + 1 < len(ir_signal_durations):
                if within_tolerance(ir_signal_durations[j], nec_bit_mark_duration):
                    if within_tolerance(ir_signal_durations[j + 1], nec_zero_space_duration):
                        nec_sequence_bits.append(0)
                    elif within_tolerance(ir_signal_durations[j + 1], nec_one_space_duration):
                        nec_sequence_bits.append(1)
                    else:
                        break
                    j += 2
                else:
                    break
            if len(nec_sequence_bits) % 8 == 0:
                return [int("".join(map(str, nec_sequence_bits[i:i + 8])), 2) for i in range(0, len(nec_sequence_bits), 8)]
    return []

# Nec sequence to Short
def decode_nec_sequence_to_short(nec_sequence_bytes: list[int]) -> str:
    """
    Decodes a NEC sequence (list of bytes) and extracts the short code (address and command).
    Returns the short code as a formatted hex string if identified, otherwise a message indicating no valid command.
    """
    if len(nec_sequence_bytes) < 4:
        return "No valid NEC command detected"

    address_byte_1, address_byte_2 = nec_sequence_bytes[:2]
    is_nec_extended = address_byte_1 != (address_byte_2 ^ 0xFF) # corrected condition

    if is_nec_extended:
        address_hex_string = f"{_reverse_bits(address_byte_1):02X} {_reverse_bits(address_byte_2):02X}"
    else:
        address_hex_string = f"{_reverse_bits(address_byte_1):02X}"

    command_bytes_hex_strings = []
    for i in range(2, len(nec_sequence_bytes) - 1, 2):
        if nec_sequence_bytes[i] == (nec_sequence_bytes[i + 1] ^ 0xFF):
            command_bytes_hex_strings.append(f"{_reverse_bits(nec_sequence_bytes[i]):02X}")
        else:
            break

    if command_bytes_hex_strings:
        command_hex_string = " ".join(command_bytes_hex_strings)
        nec_type_text = "NEC extended" if is_nec_extended else "NEC standard"
        return f"{nec_type_text} command detected : Hex :blue[{address_hex_string}] :red[{command_hex_string}]"

    return "No valid NEC command detected"

# Nec sequence to Raw
def encode_nec_sequence_to_raw(nec_sequence_bytes: list[int], repeats_count: int = 1) -> list[int]:
    """Encodes a NEC sequence (list of bytes) into a raw IR signal (list of durations)."""
    preamble_durations = [9000, 4500]
    repeat_signal_durations = [9000, 2250, 562] # NEC repeat signal timing
    zero_durations = [562, 563]
    one_durations = [562, 1688]
    trailer_duration = [562]
    raw_signal_durations = []

    for repeat_num in range(repeats_count):
        if repeat_num == 0:
            # First transmission uses standard preamble
            raw_signal_durations.extend(preamble_durations)
        else:
            # Subsequent transmissions use repeat signal
            raw_signal_durations.extend(repeat_signal_durations)

        for byte_val in nec_sequence_bytes:
            for i in range(8):  # Process each bit of the byte
                if (byte_val >> (7-i)) & 1: # Check if the i-th bit is a 1, MSB first now
                    raw_signal_durations.extend(one_durations)
                else:
                    raw_signal_durations.extend(zero_durations)

        raw_signal_durations.extend(trailer_duration) # Trailer after each sequence including repetitions

    return raw_signal_durations

# Nec sequence to Short
def encode_short_to_nec_sequence(short_code_bytes: list[int], is_nec_extended: bool) -> list[int]:
    """
    Converts a short code (address and command) to a NEC sequence (list of bytes).
    Supports NEC standard and NEC extended (NECx) formats.
    """

    if len(short_code_bytes) < 2:
        st.warning("At least one address and one command are required in a *short* string.", icon="⚠️")
        return None

    nec_sequence_bytes = []

    if is_nec_extended:
        # NEC extended (NECx)
        if len(short_code_bytes) == 2: # Handle 2-byte short code case for NECx
            address_high_byte = 0x00  # Assume high byte of address is 0x00
            address_low_byte = short_code_bytes[0]
            command_bytes = short_code_bytes[1:] # Command is the second byte
            nec_sequence_bytes.extend([_reverse_bits(address_high_byte), _reverse_bits(address_low_byte)]) # High then Low address bytes

        elif len(short_code_bytes) >= 3: # Standard NECx case with 3+ bytes
            address_high_byte = short_code_bytes[0]
            address_low_byte = short_code_bytes[1]
            nec_sequence_bytes.extend([_reverse_bits(address_high_byte), _reverse_bits(address_low_byte)]) # High then Low address bytes
            command_bytes = short_code_bytes[2:] # Commands start from the third byte

            # Check for prohibited addresses (ambiguous with NEC standard)
            if nec_sequence_bytes[0] == (nec_sequence_bytes[1] ^ 0xFF):
                st.error("This address is prohibited in NEC Extended format (inverted address bytes).", icon="⚠️")
                return None
        else: # case of len(short_code_bytes) < 2 should be already handled at the beginning, but for safety
            st.warning("For NEC extended, at least address and command are required.", icon="⚠️")
            return None


        # Process commands (common to both 2-byte and 3+ byte NECx cases)
        for command_byte in command_bytes:
            command_val = _reverse_bits(command_byte)
            nec_sequence_bytes.extend([command_val, command_val ^ 0xFF])


    else:
        # NEC standard (no change needed here)
        address_byte = short_code_bytes[0]
        nec_sequence_bytes.extend([_reverse_bits(address_byte), _reverse_bits(address_byte) ^ 0xFF])
        # Process commands (starting from the second byte in short)
        command_bytes = short_code_bytes[1:]
        for command_byte in command_bytes:
            command_val = _reverse_bits(command_byte)
            nec_sequence_bytes.extend([command_val, command_val ^ 0xFF])

    return nec_sequence_bytes

# Raw to Broadlink
def encode_raw_to_broadlink_code(signal: list[int]) -> bytes:
    """
    Encodes a raw IR signal (list of durations in µs) into Broadlink format.
    Returns the Broadlink code as bytes.
    """
    def encode_value(value: int) -> list[int]:
        encoded = math.floor(value * 269 / 8192)
        if encoded > 255:
            return [0x00, (encoded >> 8) & 0xFF, encoded & 0xFF]  # Encode as big-endian word
        else:
            return [encoded]  # Encode as a single byte

    payload = []
    for value in signal:
        payload.extend(encode_value(value))
    
    length = len(payload) + 4  # Include header bytes in length
    header = [0x26, 0x00, len(payload) & 0xFF, (len(payload) >> 8) & 0xFF]
    footer = [0x0D, 0x05]
    
    final_code = header + payload + footer
    
    # Ensure the total length is a multiple of 16 by padding with 0x00
    while len(final_code) % 16 != 0:
        final_code.append(0x00)
    
    # Convert to bytes and encode in base64
    final_bytes = bytes(final_code)
    # b64_encoded = base64.b64encode(final_bytes).decode('utf-8')
    # st.write("".join(f"{byte:02X}" for byte in final_code)) # Broadlink Hex, if needed
    
    # return b64_encoded
    return final_bytes

# Broadlink to Raw
def decode_broadlink_hex_to_raw(broadlink_hex_code: str) -> list[int]:
    """
    Decodes a Broadlink hex code string into a raw IR signal (list of durations in µs).

    Args:
        broadlink_hex_code: The Broadlink hex code string.

    Returns:
        A list of raw IR signal durations (integers).
        Raises ValueError if the hex code is invalid or not in Broadlink format.
    """
    broadlink_hex_code = broadlink_hex_code.replace(" ", "").strip() # remove spaces and strip
    if not broadlink_hex_code:
        return [] # empty input

    if not all(c in '0123456789abcdefABCDEF' for c in broadlink_hex_code):
        raise ValueError("Invalid hex code: contains non-hexadecimal characters.")


    try:
        broadlink_bytes = bytes.fromhex(broadlink_hex_code)
    except ValueError:
        raise ValueError("Invalid hex code.")

    if len(broadlink_bytes) < 16: # Broadlink codes are usually at least 16 bytes due to padding
        raise ValueError("Invalid Broadlink code: too short.")
    if broadlink_bytes[0] not in [0x26, 0xb2, 0xd7]: # Check header
        raise ValueError("Invalid Broadlink header.")


    payload_length = broadlink_bytes[2] + (broadlink_bytes[3] << 8)
    if payload_length > len(broadlink_bytes) - 6: # Header (4) + Footer (2) = 6
        raise ValueError("Invalid Broadlink code: incorrect payload length in header.")

    payload = broadlink_bytes[4:4+payload_length] # Extract payload based on length from header
    raw_signal_durations = []
    i = 0
    while i < len(payload):
        value = payload[i]
        if value == 0x00: # Big-endian word encoding
            if i + 2 >= len(payload):
                raise ValueError("Invalid Broadlink code: incomplete big-endian word.")
            encoded_value = (payload[i+1] << 8) + payload[i+2]
            i += 3
        else: # Single byte encoding
            encoded_value = value
            i += 1

        duration_us = round(encoded_value * 8192 / 269) # Reverse the encoding
        raw_signal_durations.append(duration_us)

    return raw_signal_durations

def decode_broadlink_b64_to_hex(broadlink_b64_code: str) -> str:
    """
    Decodes a Broadlink base64 code string into a Broadlink hex code string.
    """
    broadlink_b64_code = broadlink_b64_code.strip() # remove spaces and strip
    if not broadlink_b64_code:
        return "" 

    try:
        decoded_bytes = base64.b64decode(broadlink_b64_code)
    except base64.binascii.Error:
        return ""

    broadlink_hex_code = decoded_bytes.hex().lower() # Convert bytes to hex and lowercase for consistency
    return broadlink_hex_code
#
# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------
#
def _reverse_bits(byte_val: int) -> int:
    """Reverses the bits of a byte."""
    return int(f"{byte_val:08b}"[::-1], 2)

def display_raw_signal_chart(ir_signal_durations: list[int]):
    """Displays the raw IR signal (list of durations) as a line chart using Streamlit."""
    df_signal = build_signal_dataframe_for_chart(ir_signal_durations)
    if not df_signal.empty: # avoid error if df_signal is empty
        df_signal["UpperLimit"] = 1.5  # Force vertical scale
        st.line_chart(df_signal.set_index("Time"))

def display_nec_sequence_info(ir_signal_durations: list[int], sequence_type: str):
    """Detects and displays NEC sequence (list of bytes) information using Streamlit."""
    nec_sequence_bytes = decode_raw_to_nec_sequence(ir_signal_durations)
    type_msg=f"Type: {sequence_type}"
    if nec_sequence_bytes:
        nec_hex_string = " ".join(f"{byte:02X}" for byte in nec_sequence_bytes)
        nec_sequence_text=f"\n* NEC Sequence: :orange[{nec_hex_string}]"
        nec_short_code_string = decode_nec_sequence_to_short(nec_sequence_bytes)
        if nec_short_code_string:
            nec_command_text=f"\n* {nec_short_code_string}"
        else:
            nec_command_text="No NEC command identified."
        st.success(type_msg + nec_sequence_text+nec_command_text, icon="✅")
    else:
        st.success(type_msg + "\n* No NEC sequence found.", icon="✅")
        
        st.write()

def display_conversion(raw_signal_durations: list[int], sequence_type: str, display_raw_code: bool = True, display_broadlink: bool = True, display_broadlink_hex: bool = True, display_tuya_code: bool = True):

    """Displays the raw IR signal output with flexible display options (raw code, chart, sensus, broadlink code, Tuya code)."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        display_nec_sequence_info(raw_signal_durations, sequence_type)
    with col_c2:
        st.image("remote01.png")

        
    if display_tuya_code:
        tuya_code_encoded = encode_raw_to_tuya(raw_signal_durations)
        st.write("Tuya IR Code:") # Moved from sections to here
        st.code(tuya_code_encoded, language="plaintext", wrap_lines=True) # Moved from sections to here
    if display_raw_code:
        st.write("Raw IR Sequence:")
        raw_signal_string = ", ".join(map(str, raw_signal_durations))
        st.code(raw_signal_string, language="plaintext", wrap_lines=True)
    if display_broadlink or display_broadlink_hex: 
        broadlink_bytes = encode_raw_to_broadlink_code(raw_signal_durations)
        broadlink_hex="".join(f"{byte:02X}" for byte in broadlink_bytes)
        if display_broadlink_hex:
            st.write("Broadlink Code (Hex):")
            st.code(broadlink_hex, language="plaintext", wrap_lines=True)
        if display_broadlink:
            broadlink_code_b64 = base64.b64encode(broadlink_bytes).decode('ascii')
            st.write("Broadlink Code (Base64):")
            st.code(broadlink_code_b64, language="plaintext", wrap_lines=True)

    display_raw_signal_chart(raw_signal_durations)
    st.markdown(sensus, help=help_sensus)

def display_invalid_input_warning(warning_message: str):
    """Displays a Streamlit warning message for invalid user input."""
    st.warning(warning_message, icon="⚠️")

def parse_nec_user_input(code_string: str) -> list[int]:
    """
    Interprets a user-input string as a NEC/NEC Extended code and returns a list of bytes.
    Supports hex, binary, and decimal formats with various separators.
    """
    code_string = code_string.strip().lower()
    code_string = code_string.strip('[](){}\'"`') # remove brackets, quotes, parentheses

    if not code_string:
        return []

    separators_list = [',', ';', ' ', '\t'] # common separators

    # Try parsing as separated binary values first (handling missing leading zeros)
    # e.g., "10, 01, 1110"
    for separator_char in separators_list:
        if separator_char in code_string:
            string_values = [s.strip() for s in code_string.split(separator_char) if s.strip()]
            binary_bytes_separated = _parse_separated_values(string_values, base=2, valid_chars='01', needs_padding=True)
            if binary_bytes_separated: # helper function returns None if parsing fails
                return binary_bytes_separated

    # Try parsing as continuous binary (after separated binary attempts)
    # e.g., "10010011"
    binary_continuous_str = ''.join(code_string.split())
    if all(c in '01' for c in binary_continuous_str):
        if len(binary_continuous_str) % 8 == 0:
            binary_bytes_continuous = []
            for i in range(0, len(binary_continuous_str), 8):
                byte_val = int(binary_continuous_str[i:i+8], 2)
                binary_bytes_continuous.append(byte_val)
            return binary_bytes_continuous
        else:
            pass # Invalid binary string length for continuous binary - continue to next parsing attempt
#
    # Try parsing as continuous hex (general case, covers short hex cases as well)
    # e.g., "0A0B", "F1C2D3"
    is_hex_like = all(c in '0123456789abcdef' for c in ''.join(code_string.split()))
    if is_hex_like and len(''.join(code_string.split())) % 2 == 0:
        try:
            data_bytes = bytes.fromhex(''.join(code_string.split()))
            return list(data_bytes)
        except ValueError:
            pass # Not valid hex, try other formats - continue to next parsing attempt
#
    for separator_char in separators_list:
        if separator_char in code_string:
            string_values = [s.strip() for s in code_string.split(separator_char) if s.strip()]
            try:
                # Try parsing as separated decimal values
                # e.g., "10, 255, 0"
                integer_values = [int(v) for v in string_values]
                if all(0 <= val <= 255 for val in integer_values):
                    return integer_values
            except ValueError: # Catches ValueError if int(v) fails or decimal value is out of range
                pass # Not decimal or decimal out of byte range - continue to next parsing attempt. Let other formats be tested
#
            # Try parsing as separated hex values
            # e.g., "0x0A, 0xFB, 0x3C", "A, FB, 3C"
            hex_bytes_separated = _parse_separated_values(string_values, base=16, valid_chars='0123456789abcdef', needs_padding=False)
            if hex_bytes_separated: # helper function returns None if parsing fails
                return hex_bytes_separated
#
    return None # Invalid NEC/NEC Extended code format - return None if all parsing attempts fail

def _parse_separated_values(string_values, base, valid_chars, needs_padding):
    """
    Helper function to parse separated values (binary or hex).
    Returns a list of bytes if parsing is successful, None otherwise.
    """
    try:
        separated_bytes = []
        is_valid_separated_format = True # Flag to validate the whole separated format
        for val_str in string_values:
            if not all(c in valid_chars for c in val_str):
                is_valid_separated_format = False # Not valid format, invalidate separated format
                break # Exit inner loop, separated format invalid

            if needs_padding:
                val_padded = val_str.zfill(8) # Pad with leading zeros to 8 bits if necessary (binary)
            else:
                val_padded = val_str # No padding for hex

            if len(val_padded) > 8 and needs_padding: # check length after padding for binary
                is_valid_separated_format = False # Too long, invalidate separated format
                break # Exit inner loop, separated format invalid
            elif len(val_padded) > 2 and not needs_padding: # check length for hex (max 2 hex chars per byte)
                is_valid_separated_format = False # Too long for hex, invalidate separated format
                break # Exit inner loop, separated format invalid

            byte_val = int(val_padded, base)
            if 0 <= byte_val <= 255:
                separated_bytes.append(byte_val)
            else:
                is_valid_separated_format = False # Out of range, invalidate separated format
                break # Exit inner loop, separated format invalid

        if is_valid_separated_format: # Check flag after inner loop
            return separated_bytes # Return only if VALID separated format
        else:
            return None # Parsing failed for this separator and format
    except ValueError:
        return None # Conversion failed - parsing fails


def build_signal_dataframe_for_chart(ir_signal_durations: list[int]) -> pd.DataFrame:
    """Transforms the raw IR signal into a DataFrame suitable for visualization."""
    if not ir_signal_durations:
        return pd.DataFrame(columns=["Time", "Signal"])

    current_time = 0  # Accumulate time to create a time series
    data_rows = []

    for i, duration in enumerate(ir_signal_durations):
        level_val = 1 if i % 2 == 0 else 0  # Even indices are high, odd indices are low
        data_rows.append((current_time, level_val))
        current_time += duration
        data_rows.append((current_time, level_val))  # Maintain the level for the duration

    return pd.DataFrame(data_rows, columns=["Time", "Signal"])

def omni_test(sequence: str) -> tuple[str, list[int]] | str:
    """
    Analyzes an input string to identify the type of IR code sequence.
    Returns a string indicating the detected type and signal (for Broadlink codes).
    """

    if not sequence: # Check for empty sequence at the beginning
        return "Empty"

    # Clean up input sequence: remove CR, LF, Tabs and extra whitespaces and surrounding []
    sequence_stripped = ''.join(sequence.split()).rstrip('[](){}\'\" \n\r\t') # Use rstrip instead of strip, and keep whitespace stripping from split

    # Raw check: 
    #   needs to be a comma-separated list of at least 18 decimal numbers
    raw_values = re.sub(r"[\s'\"\[\]\(\)\{\}]", "", sequence_stripped) # remove spaces and brackets
    if "," in raw_values:
        number_str_list = raw_values.split(',')
        try:
            numbers = [int(x) for x in number_str_list]
            if len(numbers) >= 18: # Minimum 18 values for Raw
                return RAW
        except ValueError:
            pass # Not Raw format if conversion to int fails

    # Broadlink Hex check: 
    #   needs to be hex,to start with 26, b2, or d7 (case-insensitive), min 42 char long
    broadlink_hex_prefixes = {0x26: "IR Signal", 0xb2: "RF 433MHz Signal", 0xd7: "RF 315MHz Signal"} # Clés sont maintenant des entiers
    hex_string_no_spaces = ''.join(sequence_stripped.split()).lower()
    if all(c in '0123456789abcdefABCDEF' for c in hex_string_no_spaces):
        for prefix, signal_type in broadlink_hex_prefixes.items():
            if hex_string_no_spaces.startswith(str(hex(prefix))[2:]): # Convert prefix to string
                if len(hex_string_no_spaces) > 42:         # 42 = minimum of one encoded byte
                    return f"{BROADLINK_HEX} ({signal_type})"

    # Broadlink B64 / Tuya check: try base64 decoding
    if re.match(r'^[A-Za-z0-9+/]*={0,2}$', sequence_stripped):
        try:
            decoded_bytes = base64.b64decode(sequence_stripped)
            first_byte = decoded_bytes[0]

            if first_byte in broadlink_hex_prefixes:
                return f"{BROADLINK_B64} ({broadlink_hex_prefixes[first_byte]})"
            else:
                if _might_be_fastlz_level1(decoded_bytes):
                    return TUYA
                else:
                    st.write("omni_test: Pré-test FastLZ Level 1 échoué. Ne considère PAS comme Tuya.")

        except Exception: # Broadlink B64 or Tuya decoding failed
            pass

    # NEC check: use parse_nec_user_input to detect NEC codes and return parsed bytes
    # Safeguard: rejects if nec_bytes length exceeds 16 bytes, as long hex can easily be taken for NEC
    nec_bytes = parse_nec_user_input(sequence)
    if nec_bytes is not None:
        if len(nec_bytes) > 16:
            pass
        else:
            return NEC, nec_bytes  # Returns tuple with type and parsed bytes
    return "Unrecognized"

def omnidecode(input_string: str, code_type: str, parsed_bytes=None):
    """
    Decodes IR code based on the detected type and displays the output.
    All conversions go through Raw format before calling display signals function
    """
    if not input_string: # Exit function early if input is empty
        display_invalid_input_warning("Please enter a valid sequence to decode.")
        return

    if code_type == TUYA:
        try:
            raw_bytes_sequence = decode_tuya_to_raw(input_string)
            display_conversion(raw_bytes_sequence, code_type, display_tuya_code=False)
        except Exception as e:
            display_invalid_input_warning(f"Tuya decoding error : \n{e}")

    elif code_type == RAW:
        try:
            raw_bytes_sequence = [int(x.strip()) for x in input_string.split(',')]
            display_conversion(raw_bytes_sequence, code_type, display_raw_code=False)
        except Exception as e:
            display_invalid_input_warning(f"Raw decoding error : \n{e}")

    elif code_type.startswith(BROADLINK_HEX): # Handling Broadlink Hex type without (type comment)
        try:
            raw_bytes_sequence = decode_broadlink_hex_to_raw(input_string)
            display_conversion(raw_bytes_sequence, code_type, display_broadlink_hex=False)
        except Exception as e:
            display_invalid_input_warning(f"Hex decoding error : \n{e}")

    elif code_type.startswith(BROADLINK_B64): # Handling Broadlink B64 type type without (type comment)
        broadlink_hex=decode_broadlink_b64_to_hex(input_string)
        try:
            raw_bytes_sequence = decode_broadlink_hex_to_raw(broadlink_hex)
            display_conversion(raw_bytes_sequence, code_type, display_broadlink=False)
        except Exception as e:
            display_invalid_input_warning(f"Broadlink B64 decoding error : \n{e}")

    elif code_type == NEC:
        # Check if parsed_bytes is available
        if parsed_bytes:
            try:
                raw_bytes_sequence = encode_nec_sequence_to_raw(parsed_bytes) # Use parsed_bytes
                display_conversion(raw_bytes_sequence, code_type) 
            except Exception as e:
                display_invalid_input_warning(f"NEC encoding error : \n{e}") # Changed error message
        else:
            display_invalid_input_warning(f"NEC decoding error: Parsed bytes not available.") # Should not happen if omni_test is correct


    elif code_type == "Unrecognized":
        display_invalid_input_warning("Unrecognized IR code type. Please enter a valid code of a known type.")
    else:
        display_invalid_input_warning(f"Code type '{code_type}' not yet handled in omnidecode.") # Handling other potential types if needed

def _might_be_fastlz_level1(data: bytes) -> bool:
    """
    Checks if the data *might* be in FastLZ Level 1 compressed format.
    Heuristic check based on the first byte, FastLZ Level 1 spec.
    FastLZ documentation: https://github.com/ariya/FastLZ
    """

    if not data or len(data) < 2:
        # FastLZ Level 1 block needs at least 2 bytes: header + data
        return False

    first_byte = data[0]

    # Extract block tag (3 most significant bits - MSB)
    block_tag = first_byte >> 5

    # Check if block tag is 0 for Level 1
    if block_tag != 0b000: # Block tag for Level 1 must be 0 (binary '000')
        return False

    # Extract literal run size (5 least significant bits - LSB) and add 1
    literal_run_size = (first_byte & 0b00011111) + 1 # Mask to keep only 5 LSB

    if len(data) < literal_run_size:
        # Data length is shorter than the declared literal run size
        # Not enough bytes to contain the literal run.
        return False

    return True
#
# --------------------------------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------------------------------
#
st.header("IRTuya : IR Converter for Tuya, Broadlink, Raw, LIRC & NEC Infrared codes")
st.markdown("---")
#
# --------------------------------------------------------------------------------------------------
#  IR Transcoder
# --------------------------------------------------------------------------------------------------
#
col_a1, col_a2 = st.columns([3, 1])
with col_a1:
    st.subheader("IR Transcoder")
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        show_examples = st.checkbox("Examples", value=False)
        if show_examples:
            selected_example_with_label = st.selectbox(
                "Examples: ",
                key="example",
                options=example_sequences["Examples"],
                index=0,
                help="Examples of IR sequence formats recognized."
                )
            if selected_example_with_label:
                # Remove 'title' part of the example seqs, like "Nec: 2abf" -> "2abf"
                selected_example_seq = selected_example_with_label.split(": ", 1)[-1]
            else:
                selected_example_seq = ""
        else:
            selected_example_seq = example_sequences["Examples"][0].split(": ", 1)[-1] if example_sequences["Examples"][0] else ""

with col_a2:
    st.image("remote02.png")

code_input = st.text_area("Enter IR sequence to convert:",
    value=selected_example_seq,
    max_chars=MAX_OMNIBOX_INPUT_LENGTH,
    help="Recognized types: Tuya IR, Broadlink B64, Broadlink Hex, Raw decimal."
    )

decode_button = st.button("Convert")

# Need to check if omni_test returned a tuple (NEC case), in which case second element would
# contain the code. For other code types, omni_result is just the code_type string.

if decode_button:
    omni_result = omni_test(code_input)
    if isinstance(omni_result, tuple): 
        code_type, parsed_bytes = omni_result
        omnidecode(code_input, code_type, parsed_bytes)
    else:
        code_type = omni_result
        omnidecode(code_input, code_type)
#
st.markdown("---")
#
# --------------------------------------------------------------------------------------------------
#  NEC / LIRC Sequences & Commands coder
# --------------------------------------------------------------------------------------------------
#
col_c1, col_c2 = st.columns([3, 1])
with col_c1:
    st.subheader("Encode NEC / LIRC")
    show_nec_converter = st.checkbox("Show", value=True)
with col_c2:
    st.image("remote03.png")

if show_nec_converter:
    col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
    with col_c1:
        selected_example_code = st.selectbox("Accepted sequence examples:",
            options=example_nec_seq["Examples"], 
            index=0, 
            help="Sequence is the actual value being sent."
            )
        nec_sequence_input_str = st.text_input("Enter NEC or NECx sequence:",
            value=selected_example_code,
            max_chars=MAX_NECSEQ_INPUT_LENGTH,
            help="Enter NEC/NEC Extended sequence in hex, binary, or decimal format. Make sure *Short* field is empty when you press *Encode*"
            )
    with col_c2:
        selected_example_seq = st.selectbox("Accepted *Shorts* examples:",
            options=example_nec_com["Examples"],
            index=0,
            help="*Shorts* are the real (*Adress* + *Command*) blocks coded inside a NEC sequence."
            )
        nec_short_code_input_str = st.text_input("Or enter *Short* (address - command) code:",
            value=selected_example_seq,
            max_chars=MAX_NECCOM_INPUT_LENGTH,
            help="Enter NEC *Short* (Address - Command) in hex, binary, or decimal format (will prevail on entered NEQ sequence if you press *Encode* button)."
            )
    with col_c3:
        user_repeats = st.text_input("Repeats",
            value="1",
            max_chars=2,
            help="Number of times the command will be sent during the sequence"
            )
        nec_extended_checkbox = st.checkbox("NECx :",
            help="Check if you want to get a NEC Extended format sequence from *short*. NECx format takes the two first bytes as *address*"
            )

    encode_nec_button = st.button("Encode NEC")

    if encode_nec_button:
        nec_sequence_parsed = []

        if nec_short_code_input_str:
            nec_short_code_parsed = parse_nec_user_input(nec_short_code_input_str)
            if nec_short_code_parsed:
                nec_sequence_parsed = encode_short_to_nec_sequence(nec_short_code_parsed, nec_extended_checkbox)
        else:
            nec_sequence_parsed = parse_nec_user_input(nec_sequence_input_str)
        if nec_sequence_parsed:
            raw_signal_durations = encode_nec_sequence_to_raw(nec_sequence_parsed, int(user_repeats))
            tuya_code_encoded = encode_raw_to_tuya(raw_signal_durations)
            display_conversion(raw_signal_durations, "NEC") # Simplified call
        else:
            display_invalid_input_warning("Invalid input. Please enter a valid NEC code.")
#
st.markdown("---")
#
# --------------------------------------------------------------------------------------------------
#  Footer
# --------------------------------------------------------------------------------------------------
#
col_c1, col_c2 = st.columns([3, 1])
with col_c1:
    st.markdown(f"*:red[{license} {source}]*")
    st.markdown(f"*{message}* **{contact}**")
    st.markdown(f"*:red[{thanks}]*")
with col_c2:
    st.image("contact.png")
    st.markdown(credits, unsafe_allow_html=True)

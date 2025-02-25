import streamlit as st
import io
import base64
import pandas as pd
from bisect import bisect
from struct import pack, unpack


Help_sensus="Copy the above Raw string, then click the link to open Sensus and paste it the 'Raw' field. In 'raw Analysis' panel, you can then click 'read raw' to get detailed analysis"
sensus="[Analyze or convert this signal in Sensus IR & RF Code Converter](https://pasthev.github.io/sensus/)"
# MAIN API

def decode_ir(code: str) -> list[int]:
    '''
    Decodes an IR code string from a Tuya blaster.
    Returns the IR signal as a list of µs durations,
    with the first duration belonging to a high state.
    '''
    payload = base64.decodebytes(code.encode('ascii'))
    payload = decompress(io.BytesIO(payload))

    signal = []
    while payload:
        assert len(payload) >= 2, \
            f'garbage in decompressed payload: {payload.hex()}'
        signal.append(unpack('<H', payload[:2])[0])
        payload = payload[2:]
    return signal

def encode_ir(signal: list[int], compression_level=2) -> str:
    '''
    Encodes an IR signal (see `decode_tuya_ir`)
    into an IR code string for a Tuya blaster.
    '''
    payload = b''.join(pack('<H', t) for t in signal)
    compress(out := io.BytesIO(), payload, compression_level)
    payload = out.getvalue()
    return base64.encodebytes(payload).decode('ascii').replace('\n', '')


# DECOMPRESSION

def decompress(inf: io.FileIO) -> bytes:
    '''
    Reads a "Tuya stream" from a binary file,
    and returns the decompressed byte string.
    '''
    out = bytearray()

    while (header := inf.read(1)):
        L, D = header[0] >> 5, header[0] & 0b11111
        if not L:
            # literal block
            L = D + 1
            data = inf.read(L)
            assert len(data) == L
        else:
            # length-distance pair block
            if L == 7:
                L += inf.read(1)[0]
            L += 2
            D = (D << 8 | inf.read(1)[0]) + 1
            data = bytearray()
            while len(data) < L:
                data.extend(out[-D:][:L-len(data)])
        out.extend(data)

    return bytes(out)


# COMPRESSION

def emit_literal_blocks(out: io.FileIO, data: bytes):
    for i in range(0, len(data), 32):
        emit_literal_block(out, data[i:i+32])

def emit_literal_block(out: io.FileIO, data: bytes):
    length = len(data) - 1
    assert 0 <= length < (1 << 5)
    out.write(bytes([length]))
    out.write(data)

def emit_distance_block(out: io.FileIO, length: int, distance: int):
    distance -= 1
    assert 0 <= distance < (1 << 13)
    length -= 2
    assert length > 0
    block = bytearray()
    if length >= 7:
        assert length - 7 < (1 << 8)
        block.append(length - 7)
        length = 7
    block.insert(0, length << 5 | distance >> 8)
    block.append(distance & 0xFF)
    out.write(block)

def compress(out: io.FileIO, data: bytes, level=2):
    '''
    Takes a byte string and outputs a compressed "Tuya stream".
    Implemented compression levels:
    0 - copy over (no compression, 3.1% overhead)
    1 - eagerly use first length-distance pair found (linear)
    2 - eagerly use best length-distance pair found
    3 - optimal compression (n^3)
    '''
    if level == 0:
        return emit_literal_blocks(out, data)

    W = 2**13 # window size
    L = 255+9 # maximum length
    distance_candidates = lambda: range(1, min(pos, W) + 1)

    def find_length_for_distance(start: int) -> int:
        length = 0
        limit = min(L, len(data) - pos)
        while length < limit and data[pos + length] == data[start + length]:
            length += 1
        return length
    find_length_candidates = lambda: \
        ( (find_length_for_distance(pos - d), d) for d in distance_candidates() )
    find_length_cheap = lambda: \
        next((c for c in find_length_candidates() if c[0] >= 3), None)
    find_length_max = lambda: \
        max(find_length_candidates(), key=lambda c: (c[0], -c[1]), default=None)

    if level >= 2:
        suffixes = []; next_pos = 0
        key = lambda n: data[n:]
        find_idx = lambda n: bisect(suffixes, key(n), key=key)
        def distance_candidates():
            nonlocal next_pos
            while next_pos <= pos:
                if len(suffixes) == W:
                    suffixes.pop(find_idx(next_pos - W))
                suffixes.insert(idx := find_idx(next_pos), next_pos)
                next_pos += 1
            idxs = (idx+i for i in (+1,-1)) # try +1 first
            return (pos - suffixes[i] for i in idxs if 0 <= i < len(suffixes))

    if level <= 2:
        find_length = { 1: find_length_cheap, 2: find_length_max }[level]
        block_start = pos = 0
        while pos < len(data):
            if (c := find_length()) and c[0] >= 3:
                emit_literal_blocks(out, data[block_start:pos])
                emit_distance_block(out, c[0], c[1])
                pos += c[0]
                block_start = pos
            else:
                pos += 1
        emit_literal_blocks(out, data[block_start:pos])
        return

    # use topological sort to find shortest path
    predecessors = [(0, None, None)] + [None] * len(data)
    def put_edge(cost, length, distance):
        npos = pos + length
        cost += predecessors[pos][0]
        current = predecessors[npos]
        if not current or cost < current[0]:
            predecessors[npos] = cost, length, distance
    for pos in range(len(data)):
        if c := find_length_max():
            for l in range(3, c[0] + 1):
                put_edge(2 if l < 9 else 3, l, c[1])
        for l in range(1, min(32, len(data) - pos) + 1):
            put_edge(1 + l, l, 0)

    # reconstruct path, emit blocks
    blocks = []; pos = len(data)
    while pos > 0:
        _, length, distance = predecessors[pos]
        pos -= length
        blocks.append((pos, length, distance))
    for pos, length, distance in reversed(blocks):
        if not distance:
            emit_literal_block(out, data[pos:pos + length])
        else:
            emit_distance_block(out, length, distance)

# NEC Detection Functions
def detect_nec_code(decoded_signal: list[int]) -> list[int]:
    """
    Detects a NEC sequence in a decoded IR signal and returns the NEC sequence,
    including possible repetitions.
    """
    NEC_HEADER_LOW, NEC_HEADER_HIGH = 9000, 4500
    NEC_BIT_MARK, NEC_ZERO_SPACE, NEC_ONE_SPACE = 560, 560, 1690
    TOLERANCE = 200

    def within_tolerance(value, target):
        return target - TOLERANCE <= value <= target + TOLERANCE

    if not decoded_signal:
        return []

    for i in range(len(decoded_signal) - 1):
        if within_tolerance(decoded_signal[i], NEC_HEADER_LOW) and within_tolerance(decoded_signal[i + 1], NEC_HEADER_HIGH):
            nec_sequence = []
            j = i + 2
            while j + 1 < len(decoded_signal):
                if within_tolerance(decoded_signal[j], NEC_BIT_MARK):
                    if within_tolerance(decoded_signal[j + 1], NEC_ZERO_SPACE):
                        nec_sequence.append(0)
                    elif within_tolerance(decoded_signal[j + 1], NEC_ONE_SPACE):
                        nec_sequence.append(1)
                    else:
                        break
                    j += 2
                else:
                    break
            if len(nec_sequence) % 8 == 0:
                return [int("".join(map(str, nec_sequence[i:i + 8])), 2) for i in range(0, len(nec_sequence), 8)]
    return []

def extract_nec_command(nec_sequence: list[int]) -> str:
    """
    Extracts the NEC command from a given NEC sequence and returns it as a formatted hex string.
    """
    if len(nec_sequence) < 4:
        return "No valid NEC command detected"

    def reverse_bits(byte):
        """Reverses the bits of an 8-bit byte."""
        return int(f"{byte:08b}"[::-1], 2)

    address_1, address_2 = nec_sequence[:2]
    is_necs = address_1 == (address_2 ^ 0xFF)

    if is_necs:
        address = f"{reverse_bits(address_1):02X}"
    else:
        address = f"{reverse_bits(address_1):02X} {reverse_bits(address_2):02X}"

    command_bytes = []
    for i in range(2, len(nec_sequence) - 1, 2):
        if nec_sequence[i] == (nec_sequence[i + 1] ^ 0xFF):
            command_bytes.append(f"{reverse_bits(nec_sequence[i]):02X}")
        else:
            break

    if command_bytes:
        command_str = " ".join(command_bytes)
        nec_type = "NEC standard" if is_necs else "NEC extended"
        return f"{nec_type} command detected : Hex :blue[{address}] :red[{command_str}]"
    
    return "No valid NEC command detected"

def translated_values(decoded_signal: list[int]) -> pd.DataFrame:
    """
    Transforms the raw IR signal into a format suitable for visualization.
    Each pulse is represented as a vertical bar with a fixed height (1 or 0),
    and its width corresponds to its duration.
    """
    if not decoded_signal:
        return pd.DataFrame(columns=["Time", "Signal"])

    time = 0  # Accumulate time to create a time series
    data = []

    for i, duration in enumerate(decoded_signal):
        level = 1 if i % 2 == 0 else 0  # Even indices are high, odd indices are low
        data.append((time, level))
        time += duration
        data.append((time, level))  # Maintain the level for the duration

    return pd.DataFrame(data, columns=["Time", "Signal"])

# Streamlit App
st.header("IRTuya : Tuya IR to NEC Converter")

col_c1, col_c2 = st.columns([3, 1])	#  
with col_c1:
    st.subheader("Decode Tuya IR")
with col_c2:
    st.image("remote02.png")

ir_code_input = st.text_area("Enter Tuya IR code:", help="It should be a string like BZgjrRE/AuAXAQGDBuAFA0AB4AcTQA/gFwFAI+ALAwcQoZgj0Qg/Ag==")

decode_button = st.button("Decode Tuya")

if decode_button:
    if ir_code_input:
        try:
            decoded_signal = decode_ir(ir_code_input)
            st.write("Raw IR Sequence:")
            signal_string = ", ".join(map(str, decoded_signal))
            st.code(signal_string, wrap_lines=True)

            df_signal = translated_values(decoded_signal)
            df_signal["UpperLimit"] = 1.5  # Column to force vertical scale
            st.line_chart(df_signal.set_index("Time"))

            nec_sequence = detect_nec_code(decoded_signal)
            if nec_sequence:
                nec_hex = " ".join(f"{byte:02X}" for byte in nec_sequence)
                nec_seq=f"NEC Sequence: :orange[{nec_hex}]"
                nec_command = extract_nec_command(nec_sequence)
                if nec_command:
                    nec_com=f" - {nec_command}"
                else:
                    nec_com="No NEC command identified."
                st.success(nec_seq+nec_com, icon="✅")

            st.markdown(sensus, help=Help_sensus)
        except Exception as e:
            st.error(f"Error during IR code decoding: {e}")
    else:
        st.warning("Please enter an IR code to decode.")

st.markdown("---")

col_c1, col_c2 = st.columns([3, 1])	#  
with col_c1:
    st.subheader("Encode Tuya IR")
with col_c2:
    st.image("remote01.png")

ir_signal_input_str = st.text_area("Enter IR Raw sequence:", help= "It should be a list of µs durations in decimal values separated by commas, e.g. 9112, 4525, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575, 575...")

encode_button = st.button("Encode for Tuya IR")


if encode_button:
    if ir_signal_input_str:
        try:
            ir_signal_input_list = [int(x.strip()) for x in ir_signal_input_str.split(',')]
            encoded_code = encode_ir(ir_signal_input_list)
            st.write("Tuya IR Code encoded:")
            st.code(encoded_code)

            if ir_signal_input_list:
                df_signal = translated_values(ir_signal_input_list)
                df_signal["UpperLimit"] = 1.5  # Force vertical scale
                st.line_chart(df_signal.set_index("Time"))

                nec_sequence = detect_nec_code(ir_signal_input_list)
                if nec_sequence:
                    nec_hex = " ".join(f"{byte:02X}" for byte in nec_sequence)
                    nec_seq=f"NEC Sequence: :orange[{nec_hex}]"
                    nec_command = extract_nec_command(nec_sequence)
                    if nec_command:
                        nec_com=f" - {nec_command}"
                    else:
                        nec_com="No NEC command identified."
                    st.success(nec_seq+nec_com, icon="✅")
                else:
                    st.write("No NEC sequence found.")
            st.markdown(sensus, help=Help_sensus)
        except ValueError:
            st.error("Error: Please enter a valid list of integers separated by commas for the IR signal.")
        except Exception as e:
            st.error(f"Error during IR signal encoding: {e}")

    else:
        st.warning("Please enter an IR signal to encode.")

st.markdown("---")

credits1="[mildsunrise](https://gist.github.com/mildsunrise/1d576669b63a260d2cff35fda63ec0b5)"
credits2="""<div style="text-align: center;">
<a href="https://pasthev.github.io/" target="_blank"><i>pasthev 2025</i></a>
</div>"""
col_c1, col_c2 = st.columns([3, 1])	#  
with col_c1:
    st.markdown(f"*Many thanks to {credits1} for the Tuya encoding analysis*")
with col_c2:
    st.image("contact.png")
    st.markdown(credits2, unsafe_allow_html=True)

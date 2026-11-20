#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Helper functions for Open FUSION Toolkit (OFT) Python interfaces

@authors Chris Hansen
@date May 2023
@ingroup doxy_oft_python
'''
import os
import subprocess
import re
import numpy
from collections import OrderedDict

# Common parameters
## Vacuum magnetic permeability
mu0 = numpy.pi*4.E-7
## Electron charge
eC = 1.60217663e-19


def oft_warning(warning,details=()):
    '''! Print warning in common format

    @param warning String describing warning
    @param details List of additional details
    '''
    print("Warning: {0}".format(warning))
    for detail_line in details:
        print("         {0}".format(detail_line))


def run_shell_command(command, timeout=10, env_vars=None):
    '''! Run a shell command

    @param command Command and arguments to run
    @param timeout Timeout for command to complete [seconds]
    @param env_vars Modifications to runtime environment
    @returns `result` STDOUT, `errcode` Error code from command
    '''
    # Run shell command
    my_env = os.environ.copy()
    if env_vars is not None:
        my_env.update(env_vars)
    pid = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=my_env)
    # Wait for process to complete or timeout
    try:
        outs, _ = pid.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        pid.kill()
        outs, _ = pid.communicate()
        oft_warning("Timeout for command `{0}`".format(command))
    errcode = pid.poll()
    result = outs.decode("utf-8")
    return result, errcode


## @cond
_NML_INT = re.compile(r"[+-]?\d+")
_NML_REAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# Quoted strings (a doubled quote stands for one quote; no line breaks), and the same or a comment
_NML_STRING = r"'(?:[^'\n]|'')*'|\"(?:[^\"\n]|\"\")*\""
_NML_STRING_OR_COMMENT = re.compile(_NML_STRING + r"|!.*")
_NML_STRING_OR_COMMA = re.compile("(" + _NML_STRING + ")|,")
# Value tokens: a quoted string, a separator, or a bare token
_NML_TOKEN = re.compile(_NML_STRING + r"|,|[^\s,]+")
# Assignment target NAME or NAME(subscripts), not preceded by part of a value
_NML_KEY = re.compile(r"(?<![\w.'\"])([A-Za-z_]\w*(?:\s*\([^()=]*\))?)\s*=")
# Group start (&NAME, $NAME) and group end (/, &END, $END, or a bare & or $)
_NML_START = re.compile(r"[&$](?!END\b)([A-Za-z_]\w*)", re.IGNORECASE)
_NML_END = re.compile(r"/|[&$]END\b|[&$](?![A-Za-z_])", re.IGNORECASE)
## @endcond


def _namelist_scalar(token):
    r'''! Value of a single namelist token, read as a literal and never evaluated as code

    @param token Token text
    @result `str` for a quoted string, `int` for an integer, `float` for a real number; any other token
    raises `ValueError`
    '''
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1].replace(2*token[0], token[0])
    if _NML_INT.fullmatch(token):
        return int(token)
    if _NML_REAL.fullmatch(token):
        return float(token)
    raise ValueError("Not a namelist literal: {0}".format(token))


def _namelist_number(token):
    r'''! Real value of one entry of a namelist array, or `None` when it is not a number

    @param token Token text
    @result `float` or `None`
    '''
    if _NML_REAL.fullmatch(token):
        return float(token)
    if token[0] in "'\"":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _namelist_value(text, as_array=False):
    r'''! Value of one namelist assignment

    Numbers give an `int` or `float` for one value and a `float` array for several values or an `r*v` repeat;
    any other content gives the value text with separators turned into spaces (a single token is unquoted).

    @param text Text after the `=` up to the next assignment or the end of the group, without comments
    @param as_array Return numbers as an array even for a single value
    @result Value
    '''
    entries = []
    repeated = False
    for token in _NML_TOKEN.findall(text):
        if token == ',':
            continue
        count, star, value = token.partition('*')
        if star and _NML_INT.fullmatch(count):
            entries.extend(int(count)*[value] if value else [])
            repeated = True
        else:
            entries.append(token)
    numbers = [_namelist_number(entry) for entry in entries]
    if entries and all(number is not None for number in numbers):
        if as_array or repeated or len(entries) > 1:
            return numpy.array(numbers, dtype=float)
        try:
            return _namelist_scalar(entries[0])
        except ValueError:
            pass
    elif not entries and as_array:
        return numpy.zeros((0,))
    # Not numbers: the text, with commas outside quoted strings and double quotes removed
    lines = [line.strip() for line in text.splitlines()]
    text = ' '.join([line for line in lines if line != ''])
    text = _NML_STRING_OR_COMMA.sub(lambda m: m.group(1) if m.group(1) else ' ', text)
    text = text.replace('"', '').strip()
    if '*' in text:
        parts = text.split()
        for i, part in enumerate(parts):
            if '*' in part:
                count = part.split('*')
                if _NML_INT.fullmatch(count[0]):
                    parts[i] = ' '.join(int(count[0])*[count[1]])
        return ' '.join(parts)
    parts = text.split()
    if len(parts) == 1 and not as_array:
        try:
            return _namelist_scalar(parts[0])
        except ValueError:
            pass
    return text


def _namelist_bottom_array(rows, data_dict, nml_keys):
    r'''! Read the coil and vessel table that follows the namelist in older machine files (`b_arr=True`)

    @param rows Table lines (without line breaks, commas or comments)
    @param data_dict Namelist values; the table (`b_arr`) and coil/vessel arrays derived from it are added
    @param nml_keys Keys read from the namelist itself
    '''
    b_arr_len = len(rows)
    data_dict['b_arr'] = numpy.zeros((b_arr_len,6))

    for j in range(b_arr_len):
        try:
            b_line = numpy.array(re.split(r'(\s+)', rows[j]))
            b_line_space_idx = []
            b_line_rmv_idx = []

            for k in range(len(b_line)):
                if b_line[k].isspace():
                    b_line_space_idx.append(k)

            b_line_space_len = (lambda x:[len(k) for k in x])(b_line[b_line_space_idx])

            for k in range(len(b_line_space_idx)):
                if b_line_space_len[k] < 10: #== min(b_line_space_len):
                    b_line_rmv_idx.append(b_line_space_idx[k])
                else:
                    b_line[b_line_space_idx[k]] = 0.0

            b_line = numpy.delete(b_line, b_line_rmv_idx)

            data_dict['b_arr'][j][0:len(b_line)] = b_line

        except:
            data_dict['b_arr'][j][:] = numpy.nan

    if 'RF' not in nml_keys:
        try:
            data_dict['RF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),0]
            data_dict['ZF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),1]
            data_dict['WF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),2]
            data_dict['HF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),3]
            data_dict['AF'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),4]
            data_dict['AF2'] = data_dict['b_arr'][0:len(data_dict['TURNFC']),5]

        except:
            print('Error in determining F-coil data')

    if 'RVS' not in nml_keys:
        try:
            VS_idx_start = len(data_dict['b_arr'])-len(data_dict['RSISVS']) #len(data_dict['TURNFC'])
            VS_idx_end = len(data_dict['b_arr'])#len(data_dict['RSISVS'])+VS_idx_start

            data_dict['RVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,0]
            data_dict['ZVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,1]
            data_dict['WVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,2]
            data_dict['HVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,3]
            data_dict['AVS'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,4]
            data_dict['AVS2'] = data_dict['b_arr'][VS_idx_start:VS_idx_end,5]

            RVS_in_barr = True

        except:
            RVS_in_barr = False
            print('Error in determining vacuum vessel data')

    if 'RE' not in nml_keys:
        if data_dict['IECOIL'] == 1:
            try:
                if 'RF' in data_dict.keys():
                    EC_idx_start = len(data_dict['TURNFC'])
                else:
                    EC_idx_start = 0

                if RVS_in_barr:
                    EC_idx_end = VS_idx_start
                else:
                    EC_idx_end = len(data_dict['b_arr'])

                data_dict['RE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,0]
                data_dict['ZE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,1]
                data_dict['WE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,2]
                data_dict['HE'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,3]
                data_dict['ECID'] = data_dict['b_arr'][EC_idx_start:EC_idx_end,4]

            except:
                print('Error in determining E-coil data')


def read_fortran_namelist(file0, silent=True, b_arr=False):
    r'''Return a dictionary with the parameters in the namelist file (file0)

    All groups (`&NAME ... /`, also ended by `&END` or `$END`) are merged into one dictionary, keyed by the
    names as written (case is kept). A name assigned again in the same group takes the later value, as in
    Fortran; a name already read from an earlier group keeps the earlier group's value. Several assignments
    may share a line; text outside the groups and comments (`!` outside quoted strings) are ignored, and a
    line containing `comment:` ends the input. A file without group headers is read as one group.

    @param file0 File to read from
    @param silent If false, print dictionary entries after reading
    @param b_arr Refers to an array at the bottom of the file, if one exists
    @result Dictionary containing data from file0.
    '''
    with open(str(file0), 'r') as f:
        f_lines = f.readlines()

    # The coil and vessel table, when requested, follows the namelist terminator closest to the last assignment
    table_start = len(f_lines)
    if b_arr:
        brk_idx = [i for i, line in enumerate(f_lines) if line.find('/') >= 0]
        equal_idx = [i for i, line in enumerate(f_lines) if line.find('=') >= 0]
        table_start = brk_idx[numpy.argmin(abs(numpy.array(brk_idx)-numpy.max(equal_idx)))]

    # Input ends at the last line containing "comment:"
    end_idx = len(f_lines)
    for i, line in enumerate(f_lines):
        if line.find('comment:') >= 0:
            end_idx = i

    # Blank out comments; `masked` also blanks quoted strings, leaving only namelist syntax
    text = ''.join(f_lines[:min(table_start, end_idx)])
    clean = []
    masked = []
    pos = 0
    for m in _NML_STRING_OR_COMMENT.finditer(text):
        clean.append(text[pos:m.start()])
        masked.append(text[pos:m.start()])
        if m.group(0).startswith('!'):
            clean.append(' '*len(m.group(0)))
            masked.append(' '*len(m.group(0)))
        else:
            clean.append(m.group(0))
            masked.append('\0'*len(m.group(0)))
        pos = m.end()
    clean = ''.join(clean) + text[pos:]
    masked = ''.join(masked) + text[pos:]

    # Group bodies as (start, end) offsets
    groups = []
    m = _NML_START.search(masked)
    while m is not None:
        end = len(masked)
        stop = _NML_END.search(masked, m.end())
        if stop is not None:
            end = stop.start()
        m_next = _NML_START.search(masked, m.end())
        if m_next is not None and m_next.start() < end:
            end = m_next.start()
        groups.append((m.end(), end))
        m = _NML_START.search(masked, end)
    if len(groups) == 0:
        start = 0
        for stop in _NML_END.finditer(masked):
            groups.append((start, stop.start()))
            start = stop.end()
        groups.append((start, len(masked)))

    # Assignments in file order: (group, name, value text)
    assignments = []
    for igroup, (start, end) in enumerate(groups):
        keys = list(_NML_KEY.finditer(masked, start, end))
        for i, key in enumerate(keys):
            value_end = end
            if i < len(keys)-1:
                value_end = keys[i+1].start()
            assignments.append((igroup, key.group(1).strip(), clean[key.end():value_end]))

    data_dict = OrderedDict()
    key_group = {}
    for i, (igroup, key, value_text) in enumerate(assignments):
        if key in key_group and key_group[key] != igroup:
            continue
        # The last assignment of the file is always returned as an array (former behavior)
        as_array = (i == len(assignments)-1) and (not b_arr)
        data_dict[key] = _namelist_value(value_text, as_array)
        key_group[key] = igroup

    if b_arr:
        rows = []
        for line in f_lines[table_start+1:end_idx]:
            line = line.replace('\n', '').replace(',', ' ').replace('"', '')
            if (line.find('&') >= 0) or (line.find('/') >= 0):
                continue
            line = line.split('!')[0].strip()
            if line != '':
                rows.append(line)
        _namelist_bottom_array(rows, data_dict, list(data_dict.keys()))

    if not silent:
        for key in data_dict:
            print(key, data_dict[key])

    return data_dict

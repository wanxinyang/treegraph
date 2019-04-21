# Copyright (c) 2019, Matheus Boni Vicari, treestruct
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2019, treestruct"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "0.11"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import glob


class Header:
    
    def detect_header_limit(self, code_lines):
        for i, c in enumerate(code_lines):
            if 'import' in c:
                return (i-1)
    
    def update_header(self, pyfile, new_header):
        with open(pyfile, 'r') as f:
            code_string = f.read()

        code_lines = code_string.split('\n')
            
        header_max_row = self.detect_header_limit(code_lines)
        
        post_header = ''
        for c in code_lines[header_max_row + 1:]:
            post_header = post_header = post_header + c + '\n'
              
        output_str = new_header + post_header
        
        with open(pyfile, 'w') as f:
            f.writelines(output_str)


if __name__ == "__main__":
    
    with open('header.txt') as f:
        new_header = f.read()
    
    pyfiles = glob.glob('../src/*.py')
    for p in pyfiles:
        h = Header()
        h.update_header(p, new_header)
    

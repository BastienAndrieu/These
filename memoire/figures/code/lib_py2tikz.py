##############
def bezier_path_to_string(path):
    s = ''
    for i, curve in enumerate(path):
        b = curve.x
        if b.shape[0] < 2: continue
        if b.shape[0] > 4: continue
        if b.shape[0] < 4:
            # convert into cubic curve (perform degree elevation)...
            print 'bezier_path_to_string: !!!'
        s += '(%s, %s) .. controls (%s,%s) and (%s,%s) .. (%s,%s) --\n' % (b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1])
    s += 'cycle'
    return s
##############
def closed_polyline_to_string(poly):
    s = ''
    for xy in poly:
        for p in xy[:-1]:
            s += '(%s, %s) -- ' % (p[0], p[1])
    s += 'cycle'
    return s
##############

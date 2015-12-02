function s = labelToString( label )

switch label
    case 1
        s = 'Airplane';
    case 2
        s = 'Car';
    case 3
        s = 'Horse';
    case 4
        s = 'None';
    otherwise
        warning('Unexpected label.')
end

end


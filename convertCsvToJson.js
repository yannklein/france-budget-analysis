const fs = require('fs');
const path = require('path');

function csvToJson(csvFilePath) {
    const csvData = fs.readFileSync(csvFilePath, 'utf8');
    const lines = csvData.split('\n').filter(line => line.trim() !== '');
    const headers = lines[0].split(',');

    const json = lines.slice(1).map(line => {
        const values = line.split(',');
        return headers.reduce((acc, header, index) => {
            acc[header] = values[index] || null;
            return acc;
        }, {});
    });

    return json;
}

// Example usage:
const csvFilePath = path.join(__dirname, 'account_name.csv');
const jsonFilePath = path.join(__dirname, 'account_name.json');
const jsonOutput = csvToJson(csvFilePath);

fs.writeFileSync(jsonFilePath, JSON.stringify(jsonOutput, null, 2), 'utf8');
console.log(`JSON data has been saved to ${jsonFilePath}`);
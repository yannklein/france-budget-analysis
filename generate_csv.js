const fs = require('fs');
const path = require('path');

const inputFile = path.join(__dirname, 'account_name.txt');
const outputFile = path.join(__dirname, 'account_name.csv');

// Read the input file
const data = fs.readFileSync(inputFile, 'utf8').split('\n');

// Initialize the CSV content
const headers =
  'class,class_id,category_1,category_1_id,category_2,category_2_id,category_3,category_3_id,category_4,category_4_id\n';
let csvContent = headers;

let className = '',
  classId = '',
  category1 = '',
  category1Id = '',
  category2 = '',
  category2Id = '',
  category3 = '',
  category3Id = '',
  category4 = '',
  category4Id = '';

data.forEach((line) => {
  const class_match = line.match(/^Classe (\d) : (.+)$/);
  const match = line.match(/^(\d+) - (.+)$/); // Match hierarchical structure
  if (class_match) {
    const [_, id, name] = class_match;
    classId = id;
    className = name;
    category1 =
      category1Id =
      category2 =
      category2Id =
      category3 =
      category3Id =
      category4 =
      category4Id =
        ''; // Reset lower levels
  } else if (match) {
    const [_, id, name] = match;
    if (id.length === 2) {
      category1Id = id;
      category1 = name;
      category2 =
        category2Id =
        category3 =
        category3Id =
        category4 =
        category4Id =
          ''; // Reset lower levels
    } else if (id.length === 3) {
      category2Id = id;
      category2 = name;
      category3 = category3Id = category4 = category4Id = ''; // Reset lower levels
    } else if (id.length === 4) {
      category3Id = id;
      category3 = name;
      category4 = category4Id = '';
    } else if (id.length === 5) {
      category4Id = id;
      category4 = name;
    }
    csvContent += `${classId},${className},${category1Id},${category1},${category2Id},${category2},${category3Id},${category3},${category4Id},${category4}\n`;
  }
});

// Write the output CSV file
fs.writeFileSync(outputFile, csvContent, 'utf8');
console.log('CSV file generated successfully.');

function doPost(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Orders') || ss.getSheets()[0];
  var data = JSON.parse(e.postData.contents);

  sheet.appendRow([
    new Date(),
    data.customer_name || '',
    data.facebook_psid || '',
    data.addressing_style || '',
    data.product_name || '',
    data.product_price || '',
    data.product_image_url || '',
    data.recommended_size || '',
    data.phone || '',
    data.address || '',
    data.delivery_zone || '',
    data.delivery_eta || '',
    data.delivery_charge || '',
    data.free_delivery ? 'TRUE' : 'FALSE',
    data.order_stage || ''
  ]);

  return ContentService
    .createTextOutput(JSON.stringify({ status: 'ok' }))
    .setMimeType(ContentService.MimeType.JSON);
}

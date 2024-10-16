use core_foundation_sys::runloop::{CFRunLoopGetMain, CFRunLoopRun, CFRunLoopStop};
use objc::{
    class,
    declare::ClassDecl,
    msg_send,
    runtime::{Class, Object, Sel},
    sel, sel_impl,
};

pub type SearchData = (String, String, String);
pub type SearchLocalFileResult<T> = Result<T, SearchLocalFileError>;

#[derive(Debug)]
pub enum SearchLocalFileError {
    #[allow(unused)]
    CreateCString(String),
}

/// 요청했던 `NSMetadataQuery` 객체를 저장하는 전역 변수
static mut QUERY: Option<*mut Object> = None;

/// `query` 문자열로 파일 시스템의 인덱스를 검색하는 함수
pub fn search_local_files_by_query(
    query_file_name: String,
) -> SearchLocalFileResult<Vec<SearchData>> {
    let query = unsafe { create_search_query(query_file_name)? };
    let query = unsafe { connect_notification_center(query) };
    let query = unsafe { run_query(query) };
    unsafe { process_query_data(query, Some(20)) }
}

/// Spotlight 쿼리를 생성하는 함수
unsafe fn create_search_query(query_file_name: String) -> SearchLocalFileResult<*mut Object> {
    // `NSMetadataQuery` 객체 생성
    let query: *mut Object = msg_send![class!(NSMetadataQuery), alloc];
    let query: *mut Object = msg_send![query, init];

    let predicate_format: *mut Object = {
        // https://developer.apple.com/library/archive/documentation/Carbon/Conceptual/SpotlightQuery/Concepts/QueryFormat.html#//apple_ref/doc/uid/TP40001849
        let c_string =
            std::ffi::CString::new(format!("(kMDItemFSName == '*.pdf' || kMDItemFSName == '*.png' || kMDItemFSName == '*.jpeg') && kMDItemDisplayName == \"*{}*\"cd", query_file_name))
                .map_err(|e| SearchLocalFileError::CreateCString(e.to_string()))?;
        msg_send![class!(NSString), stringWithUTF8String: c_string.as_ptr()]
    };
    let predicate: *mut Object =
        msg_send![class!(NSPredicate), predicateFromMetadataQueryString: predicate_format];
    // `NSMetadataQuery`에 predicate 설정
    let _: () = msg_send![query, setPredicate: predicate];

    Ok(query)
}

/// `NSMetadataQuery` 객체에 `NSNotificationCenter`를 연결하는 함수
unsafe fn connect_notification_center(query: *mut Object) -> *mut Object {
    let observer_class = Class::get("QueryDelegate").unwrap_or_else(|| {
        let mut decl = ClassDecl::new("QueryDelegate", class!(NSObject)).unwrap();
        decl.add_method(
            sel!(queryDidFinishGathering:),
            query_did_finish_gathering as extern "C" fn(&Object, Sel, *mut Object),
        );
        decl.register()
    });

    let observer_instance: *mut Object = msg_send![observer_class, new];
    let notification_center: *mut Object = msg_send![class!(NSNotificationCenter), defaultCenter];
    let _: () = msg_send![notification_center, addObserver: observer_instance
        selector: sel!(queryDidFinishGathering:)
        name: NSMetadataQueryDidFinishGatheringNotification
        object: query
    ];

    query
}

/// `NSMetadataQuery` 객체를 실행하는 함수
unsafe fn run_query(query: *mut Object) -> *mut Object {
    let previous_query = QUERY.replace(query);
    if let Some(previous_query) = previous_query {
        stop_query(previous_query);
    }
    let _ = QUERY.insert(query);
    // `NSMetadataQuery` 시작
    let _: () = msg_send![query, startQuery];
    CFRunLoopRun();
    query
}

/// `NSMetadataQuery` 객체의 결과를 처리하는 함수
unsafe fn process_query_data(
    query: *mut Object,
    max_count: Option<usize>,
) -> SearchLocalFileResult<Vec<SearchData>> {
    // 검색 완료된 결과 리스트를 가져옴
    let results: *mut Object = msg_send![query, results];
    // 결과 수 가져오기
    let result_count: usize = msg_send![results, count];
    let result_count = max_count.map_or(result_count, |max| max.min(result_count)); // 지정한 최대 갯수만큼 결과를 가져옴

    // 결과를 Vec<String>으로 저장
    let mut result_vec = Vec::new();
    for i in 0..result_count {
        let result: *mut Object = msg_send![results, objectAtIndex: i];
        let file_name: *mut Object = msg_send![result, valueForAttribute: NSMetadataItemFSNameKey];
        let path: *mut Object = msg_send![result, valueForAttribute: NSMetadataItemPathKey];
        let modified_date: *mut Object =
            msg_send![result, valueForAttribute: NSMetadataItemAttributeChangeDateKey];

        let file_name = nsstring_to_string(file_name);
        let path = nsstring_to_string(path);
        let modified_date = nsdate_to_string(modified_date)?;
        result_vec.push((file_name, path, modified_date));
    }

    Ok(result_vec)
}

/// `NSString` 객체를 `String`으로 변환하는 함수
unsafe fn nsstring_to_string(string: *mut Object) -> String {
    let utf8_string: *const i8 = msg_send![string, UTF8String];
    let c_str = std::ffi::CStr::from_ptr(utf8_string);
    let rust_str = c_str.to_string_lossy().into_owned();
    rust_str
}

/// `NSDate` 객체를 `String`으로 변환하는 함수
unsafe fn nsdate_to_string(date: *mut Object) -> SearchLocalFileResult<String> {
    let date_formatter: *mut Object = msg_send![class!(NSDateFormatter), new];
    let format_ns_string: *mut Object = {
        let c_string = std::ffi::CString::new("yyyy-MM-dd HH:mm:ss")
            .map_err(|e| SearchLocalFileError::CreateCString(e.to_string()))?;
        msg_send![class!(NSString), stringWithUTF8String: c_string.as_ptr()]
    };
    let _: () = msg_send![date_formatter, setDateFormat: format_ns_string];
    let ns_string: *mut Object = msg_send![date_formatter, stringFromDate: date];
    let utf8_string: *const i8 = msg_send![ns_string, UTF8String];
    let c_str = std::ffi::CStr::from_ptr(utf8_string);
    let rust_str = c_str.to_string_lossy().into_owned();
    let _: () = msg_send![date_formatter, release];
    Ok(rust_str)
}

unsafe fn stop_query(query: *mut Object) {
    let _: () = msg_send![query, stopQuery];
    CFRunLoopStop(CFRunLoopGetMain());
}

extern "C" {
    // `NSMetadataQueryDidFinishGatheringNotification`은 NSString* 타입으로 정의됨
    static NSMetadataQueryDidFinishGatheringNotification: *const Object;

    static NSMetadataItemFSNameKey: *const Object;
    static NSMetadataItemPathKey: *const Object;
    static NSMetadataItemAttributeChangeDateKey: *const Object;
}

extern "C" fn query_did_finish_gathering(_this: &Object, _: Sel, _notification: *mut Object) {
    // 알림 발생 후 RunLoop 종료
    unsafe {
        if let Some(query) = QUERY.take() {
            stop_query(query);
        }
    }
}
